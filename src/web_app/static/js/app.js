// Main application JavaScript

let currentFile = null;
let currentResults = null;

// Upload area drag and drop
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

async function handleFile(file) {
    if (!file.name.match(/\.(csv|xlsx|xls)$/i)) {
        alert('يرجى اختيار ملف CSV أو Excel');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            currentFile = data.filename;
            displayFileInfo(data);
            fileInfo.style.display = 'block';
            
            // Populate target column dropdown
            const targetColumnSelect = document.getElementById('targetColumn');
            targetColumnSelect.innerHTML = '';
            const option = document.createElement('option');
            option.value = data.target_column;
            option.textContent = data.target_column + ' (مُستنتج تلقائياً)';
            option.selected = true;
            targetColumnSelect.appendChild(option);
        } else {
            alert('خطأ: ' + data.error);
        }
    } catch (error) {
        alert('خطأ في رفع الملف: ' + error.message);
    }
}

function displayFileInfo(data) {
    document.getElementById('fileName').textContent = data.filename;
    document.getElementById('nSamples').textContent = data.n_samples;
    document.getElementById('nFeatures').textContent = data.n_features;
}

// Run analysis
document.getElementById('runAnalysisBtn').addEventListener('click', async () => {
    if (!currentFile) {
        alert('يرجى رفع ملف أولاً');
        return;
    }

    const targetColumn = document.getElementById('targetColumn').value;
    const populationSize = parseInt(document.getElementById('populationSize').value);
    const nGenerations = parseInt(document.getElementById('nGenerations').value);

    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');
    
    loadingIndicator.style.display = 'block';
    resultsSection.style.display = 'none';
    
    // Show progress bar
    showProgressBar(loadingIndicator);

    try {
        const response = await fetch('/api/run_analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: currentFile,
                target_column: targetColumn,
                ga_params: {
                    population_size: populationSize,
                    n_generations: nGenerations
                }
            })
        });

        const data = await response.json();

        if (data.success && data.job_id) {
            // Poll for progress
            pollProgress(data.job_id, loadingIndicator, resultsSection);
        } else {
            alert('خطأ: ' + (data.error || 'Unknown error'));
            loadingIndicator.style.display = 'none';
        }
    } catch (error) {
        alert('خطأ في التحليل: ' + error.message);
        loadingIndicator.style.display = 'none';
    }
});

function showProgressBar(container) {
    container.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">جاري التحميل...</span>
            </div>
            <div class="progress mb-2" style="height: 25px;">
                <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated" 
                     role="progressbar" style="width: 0%">0%</div>
            </div>
            <p id="progressMessage" class="mt-2">بدء التحليل...</p>
            <div id="progressDetails" class="small text-muted"></div>
        </div>
    `;
}

function updateProgressBar(progress, message, details) {
    const progressBar = document.getElementById('progressBar');
    const progressMessage = document.getElementById('progressMessage');
    const progressDetails = document.getElementById('progressDetails');
    
    if (progressBar) {
        progressBar.style.width = progress + '%';
        progressBar.textContent = Math.round(progress) + '%';
    }
    
    if (progressMessage) {
        progressMessage.textContent = message || 'جاري المعالجة...';
    }
    
    if (progressDetails && details) {
        let detailsHTML = '';
        if (details.current_method) {
            detailsHTML += `<strong>الطريقة الحالية:</strong> ${details.current_method}<br>`;
        }
        if (details.best_fitness !== undefined) {
            detailsHTML += `<strong>Fitness:</strong> ${details.best_fitness} | `;
        }
        if (details.n_features !== undefined) {
            detailsHTML += `<strong>الميزات:</strong> ${details.n_features} | `;
        }
        if (details.model_score !== undefined) {
            detailsHTML += `<strong>النتيجة:</strong> ${details.model_score}`;
        }
        progressDetails.innerHTML = detailsHTML;
    }
}

function pollProgress(jobId, loadingIndicator, resultsSection) {
    let pollCount = 0;
    const maxPolls = 600; // Max 5 minutes (600 * 500ms)
    
    const pollInterval = setInterval(async () => {
        pollCount++;
        
        // Timeout after max polls
        if (pollCount > maxPolls) {
            clearInterval(pollInterval);
            alert('انتهت مهلة التحليل. يرجى المحاولة مرة أخرى.');
            loadingIndicator.style.display = 'none';
            return;
        }
        
        try {
            const response = await fetch(`/api/progress/${jobId}`);
            
            // Check if response is OK
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', response.status, errorText);
                throw new Error(`Server error: ${response.status}`);
            }
            
            // Check content type
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                console.error('Invalid content type:', contentType);
                throw new Error('Server returned non-JSON response');
            }
            
            const data = await response.json();
            
            // Handle errors
            if (data.error || data.status === 'error') {
                clearInterval(pollInterval);
                alert('خطأ: ' + (data.message || data.error || 'Unknown error'));
                loadingIndicator.style.display = 'none';
                return;
            }
            
            // Update progress bar
            updateProgressBar(
                data.progress || 0,
                data.message || 'جاري المعالجة...',
                {
                    current_method: data.current_method,
                    best_fitness: data.best_fitness,
                    n_features: data.n_features,
                    model_score: data.model_score
                }
            );
            
            // Check if complete
            if (data.status === 'complete') {
                clearInterval(pollInterval);
                if (data.results && data.results.results) {
                    currentResults = data.results;
                    displayResults(data.results.results);
                } else {
                    alert('تم الانتهاء لكن النتائج غير متوفرة');
                }
                loadingIndicator.style.display = 'none';
                resultsSection.style.display = 'block';
            } else if (data.status === 'not_found') {
                clearInterval(pollInterval);
                alert('المهمة غير موجودة. يرجى المحاولة مرة أخرى.');
                loadingIndicator.style.display = 'none';
            }
        } catch (error) {
            console.error('Poll error:', error);
            // Don't stop polling immediately, might be temporary network issue
            if (pollCount > 10) { // Only show error after 10 failed attempts
                clearInterval(pollInterval);
                alert('خطأ في متابعة التقدم: ' + error.message);
                loadingIndicator.style.display = 'none';
            }
        }
    }, 500); // Poll every 500ms
}

function displayResults(results) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';

    // Create comparison table
    let tableHTML = `
        <div class="table-responsive">
            <table class="table table-striped table-hover">
                <thead class="table-dark">
                    <tr>
                        <th>الطريقة</th>
                        <th>عدد الميزات</th>
                        <th>وقت التشغيل (ثانية)</th>
                        <th>دقة الاختبار</th>
                        <th>F1 Score</th>
                        <th>CV Score</th>
                    </tr>
                </thead>
                <tbody>
    `;

    for (const [key, result] of Object.entries(results)) {
        tableHTML += `
            <tr>
                <td><strong>${result.method}</strong></td>
                <td>${result.n_features}</td>
                <td>${result.fit_time}</td>
                <td>${(result.test_accuracy * 100).toFixed(2)}%</td>
                <td>${result.test_f1.toFixed(4)}</td>
                <td>${result.cv_score.toFixed(4)}</td>
            </tr>
        `;
    }

    tableHTML += `
                </tbody>
            </table>
        </div>
    `;

    container.innerHTML = tableHTML;

    // Display method cards with feature details
    for (const [key, result] of Object.entries(results)) {
        const methodCard = document.createElement('div');
        methodCard.className = 'method-card';
        methodCard.innerHTML = `
            <h5>${result.method}</h5>
            <div class="row">
                <div class="col-md-3">
                    <strong>عدد الميزات المختارة:</strong> ${result.n_features}
                </div>
                <div class="col-md-3">
                    <strong>دقة الاختبار:</strong> ${(result.test_accuracy * 100).toFixed(2)}%
                </div>
                <div class="col-md-3">
                    <strong>F1 Score:</strong> ${result.test_f1.toFixed(4)}
                </div>
                <div class="col-md-3">
                    <strong>وقت التشغيل:</strong> ${result.fit_time} ثانية
                </div>
            </div>
            <div class="mt-3">
                <strong>الميزات المختارة:</strong>
                <div class="feature-list">
                    ${result.feature_names.map(f => `<span class="badge bg-primary badge-feature">${f}</span>`).join('')}
                </div>
            </div>
        `;
        container.appendChild(methodCard);
    }

    // Create comparison chart
    createComparisonChart(results);

    // Add GA evolution chart if available
    if (results.genetic_algorithm && results.genetic_algorithm.history) {
        createEvolutionChart(results.genetic_algorithm.history);
    }
}

function createComparisonChart(results) {
    const methods = Object.values(results).map(r => r.method);
    const accuracies = Object.values(results).map(r => r.test_accuracy * 100);
    const f1Scores = Object.values(results).map(r => r.test_f1);
    const nFeatures = Object.values(results).map(r => r.n_features);

    const trace1 = {
        x: methods,
        y: accuracies,
        name: 'دقة الاختبار (%)',
        type: 'bar',
        marker: { color: '#667eea' }
    };

    const trace2 = {
        x: methods,
        y: nFeatures,
        name: 'عدد الميزات',
        type: 'bar',
        yaxis: 'y2',
        marker: { color: '#764ba2' }
    };

    const layout = {
        title: 'مقارنة الطرق',
        xaxis: { title: 'الطريقة' },
        yaxis: { title: 'دقة الاختبار (%)' },
        yaxis2: {
            title: 'عدد الميزات',
            overlaying: 'y',
            side: 'right'
        },
        barmode: 'group'
    };

    Plotly.newPlot('comparisonChart', [trace1, trace2], layout);
}

function createEvolutionChart(history) {
    const evolutionDiv = document.createElement('div');
    evolutionDiv.id = 'evolutionChart';
    evolutionDiv.className = 'mt-4';
    document.getElementById('resultsContainer').appendChild(evolutionDiv);

    const trace = {
        x: history.best_fitness.map((_, i) => i + 1),
        y: history.best_fitness,
        name: 'أفضل Fitness',
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: '#667eea' }
    };

    const trace2 = {
        x: history.avg_fitness.map((_, i) => i + 1),
        y: history.avg_fitness,
        name: 'متوسط Fitness',
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: '#764ba2' }
    };

    const layout = {
        title: 'تطور الخوارزمية الجينية عبر الأجيال',
        xaxis: { title: 'الجيل' },
        yaxis: { title: 'Fitness Score' }
    };

    Plotly.newPlot('evolutionChart', [trace, trace2], layout);
}

