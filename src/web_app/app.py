"""
Flask web application for feature selection
"""

import os
import json
import numpy as np
import pandas as pd
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pickle
from datetime import datetime
import threading
import uuid

from ..data_preprocessing import DataPreprocessor
from ..genetic_algorithm import GeneticAlgorithm
from ..compare_methods import MethodComparer
from ..traditional_methods import (
    CorrelationSelector,
    MutualInfoSelector,
    UnivariateSelector,
    RecursiveEliminationSelector,
    PCASelector
)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Progress tracking (in-memory)
progress_storage = {}


def _to_json_safe(value):
    """Recursively convert numpy types and arrays to JSON-serializable python types."""
    import numpy as _np
    import pandas as _pd
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (_np.generic,)):
        return value.item()
    if isinstance(value, (_np.ndarray, list, tuple, set)):
        return [ _to_json_safe(v) for v in list(value) ]
    if isinstance(value, dict):
        return { str(k): _to_json_safe(v) for k, v in value.items() }
    if isinstance(value, (_pd.Series,)):
        return _to_json_safe(value.tolist())
    if isinstance(value, (_pd.DataFrame,)):
        return _to_json_safe(value.to_dict(orient='records'))
    return str(value)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataset(filepath):
    """Load dataset from file."""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, encoding='utf-8')
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, engine='openpyxl')
        elif filepath.endswith('.xls'):
            df = pd.read_excel(filepath, engine='xlrd')
        else:
            raise ValueError("Unsupported file format")
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        return df
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    target_column = request.form.get('target_column', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and validate dataset
            df = load_dataset(filepath)
            
            # Validate target column
            if not target_column or target_column not in df.columns:
                # Try to infer target (usually last column)
                target_column = df.columns[-1]
            
            # Get feature columns
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns].values
            y = df[target_column].values
            
            # Return dataset info
            return jsonify({
                'success': True,
                'filename': filename,
                'n_samples': len(df),
                'n_features': len(feature_columns),
                'feature_columns': feature_columns,
                'target_column': target_column,
                'target_unique': int(len(np.unique(y)))
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400


def run_analysis_threaded(job_id, filename, target_column, ga_params):
    """Run analysis in a separate thread with progress tracking."""
    global progress_storage
    
    try:
        # Initialize progress
        progress_storage[job_id] = {
            'status': 'initializing',
            'progress': 0,
            'message': 'Initializing...'
        }
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Update progress
        progress_storage[job_id] = {
            'status': 'loading',
            'progress': 0,
            'message': 'Loading dataset...'
        }
        
        # Load dataset
        df = load_dataset(filepath)
        feature_columns = [col for col in df.columns if col != target_column]
        X_raw = df[feature_columns]
        y_raw = df[target_column]
        
        progress_storage[job_id] = {
            'status': 'preprocessing',
            'progress': 10,
            'message': 'Preprocessing data...'
        }
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(X_raw, y_raw)
        
        progress_storage[job_id] = {
            'status': 'running',
            'progress': 20,
            'message': 'Starting analysis...',
            'current_method': None
        }
        
        # Create progress callback for GA
        def ga_progress_callback(data):
            progress = 20 + (data['progress'] * 0.4)  # GA takes 40% of total progress
            progress_storage[job_id] = {
                'status': 'running',
                'progress': progress,
                'message': f"Genetic Algorithm - Generation {data['generation']}/{data['total_generations']}",
                'current_method': 'Genetic Algorithm',
                'best_fitness': round(data['best_fitness'], 4),
                'n_features': data['best_n_features'],
                'model_score': round(data['model_score'], 4)
            }
        
        # Add progress callback to ga_params
        ga_params['progress_callback'] = ga_progress_callback
        
        # Run comparison with progress updates
        comparer = MethodComparer(X, y, random_state=42)
        
        # Update progress for each method
        methods_list = [
            ('genetic_algorithm', 'Genetic Algorithm', 20, 60),
            ('correlation', 'Correlation', 60, 70),
            ('mutual_information', 'Mutual Information', 70, 80),
            ('univariate', 'Univariate', 80, 90),
            ('rfe', 'RFE', 90, 95),
            ('pca', 'PCA', 95, 100)
        ]
        
        results = {}
        for method_key, method_name, start_progress, end_progress in methods_list:
            progress_storage[job_id] = {
                'status': 'running',
                'progress': start_progress,
                'message': f'Running {method_name}...',
                'current_method': method_name
            }
            
            if method_key == 'genetic_algorithm':
                # GA has its own progress callback already set
                start_time = time.time()
                ga = GeneticAlgorithm(X, y, **ga_params)
                ga_result = ga.run(verbose=False)
                ga_time = time.time() - start_time
                
                results['genetic_algorithm'] = comparer.evaluate_method(
                    'Genetic Algorithm',
                    ga_result['selected_features'],
                    ga_time
                )
                results['genetic_algorithm']['history'] = ga_result['history']
            else:
                # Traditional methods
                start_time = time.time()
                
                if method_key == 'correlation':
                    from ..traditional_methods import CorrelationSelector
                    selector = CorrelationSelector(k=min(20, X.shape[1] // 2))
                    selector.fit(X, y)
                    selected = selector.get_selected_features()
                elif method_key == 'mutual_information':
                    from ..traditional_methods import MutualInfoSelector
                    selector = MutualInfoSelector(k=min(20, X.shape[1] // 2))
                    selector.fit(X, y)
                    selected = selector.get_selected_features()
                elif method_key == 'univariate':
                    from ..traditional_methods import UnivariateSelector
                    selector = UnivariateSelector(k=min(20, X.shape[1] // 2))
                    selector.fit(X, y)
                    selected = selector.get_selected_features()
                elif method_key == 'rfe':
                    from ..traditional_methods import RecursiveEliminationSelector
                    selector = RecursiveEliminationSelector(n_features_to_select=min(20, X.shape[1] // 2))
                    selector.fit(X, y)
                    selected = selector.get_selected_features()
                elif method_key == 'pca':
                    from ..traditional_methods import PCASelector
                    selector = PCASelector(n_components=0.95)
                    selector.fit(X, y)
                    selected = selector.get_selected_features()
                
                method_time = time.time() - start_time
                results[method_key] = comparer.evaluate_method(method_name, selected, method_time)
            
            progress_storage[job_id] = {
                'status': 'running',
                'progress': end_progress,
                'message': f'Completed {method_name}',
                'current_method': method_name
            }
        
        # Prepare response
        response_data = {
            'success': True,
            'results': {}
        }
        
        for method_name, result in results.items():
            response_data['results'][method_name] = {
                'method': result['method'],
                'n_features': result['n_features'],
                'fit_time': round(result['fit_time'], 3),
                'test_accuracy': round(result['test_accuracy'], 4),
                'test_f1': round(result['test_f1'], 4),
                'cv_score': round(result['cv_score'], 4),
                'selected_features': result['selected_features'],
                'feature_names': [feature_columns[i] for i in result['selected_features']]
            }
            
            # Add history for genetic algorithm
            if method_name == 'genetic_algorithm' and 'history' in result:
                response_data['results'][method_name]['history'] = {
                    'best_fitness': [round(f, 4) for f in result['history']['best_fitness']],
                    'avg_fitness': [round(f, 4) for f in result['history']['avg_fitness']],
                    'best_n_features': result['history']['best_n_features'],
                    'generation_time': [round(t, 3) for t in result['history']['generation_time']]
                }
        
        progress_storage[job_id] = {
            'status': 'complete',
            'progress': 100,
            'message': 'Analysis complete!',
            'results': _to_json_safe(response_data)
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        progress_storage[job_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}',
            'error_details': error_trace
        }


@app.route('/api/run_analysis', methods=['POST'])
def run_analysis():
    """Run feature selection analysis (async)."""
    data = request.json
    filename = data.get('filename')
    target_column = data.get('target_column')
    ga_params = data.get('ga_params', {})
    
    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Start analysis in background thread
    thread = threading.Thread(
        target=run_analysis_threaded,
        args=(job_id, filename, target_column, ga_params)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Analysis started'
    })


@app.route('/api/progress/<job_id>')
def get_progress(job_id):
    """Get progress of running analysis."""
    try:
        if job_id not in progress_storage:
            return jsonify({
                'status': 'not_found',
                'error': 'Job not found',
                'progress': 0
            }), 404
        
        progress_data = progress_storage[job_id].copy()
        
        # If complete, clean up after returning
        if progress_data.get('status') == 'complete':
            results = progress_data.pop('results', {})
            # Clean up old progress after 1 minute
            return jsonify({
                'status': 'complete',
                'progress': 100,
                'message': 'Analysis complete!',
                'results': results
            })
        
        # Return current progress
        return jsonify(_to_json_safe({
            'status': progress_data.get('status', 'unknown'),
            'progress': progress_data.get('progress', 0),
            'message': progress_data.get('message', 'Processing...'),
            'current_method': progress_data.get('current_method'),
            'best_fitness': progress_data.get('best_fitness'),
            'n_features': progress_data.get('n_features'),
            'model_score': progress_data.get('model_score')
        }))
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'progress': 0
        }), 500


@app.route('/api/results/<result_id>')
def get_results(result_id):
    """Get saved results."""
    result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_id)
    
    try:
        with open(result_filepath, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    except FileNotFoundError:
        return jsonify({'error': 'Results not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

