/**
 * Medical Insurance Cost Predictor - JavaScript
 * Handles form submission, API calls, and UI updates
 */

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const resultsContainer = document.getElementById('resultsContainer');
const predictionAmount = document.getElementById('predictionAmount');
const insightsList = document.getElementById('insightsList');
const inputSummary = document.getElementById('inputSummary');

// Form submission handler
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Get form data
    const formData = {
        age: parseInt(document.getElementById('age').value),
        sex: document.getElementById('sex').value,
        bmi: parseFloat(document.getElementById('bmi').value),
        children: parseInt(document.getElementById('children').value),
        smoker: document.getElementById('smoker').value,
        region: document.getElementById('region').value
    };

    // Validate form data
    if (!validateFormData(formData)) {
        return;
    }

    // Show loading state
    showLoading();

    try {
        // Make API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Prediction failed');
        }

        const result = await response.json();

        // Display results
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    } finally {
        hideLoading();
    }
});

/**
 * Validate form data
 */
function validateFormData(data) {
    // Age validation
    if (data.age < 18 || data.age > 100) {
        showError('Age must be between 18 and 100');
        return false;
    }

    // BMI validation
    if (data.bmi < 10 || data.bmi > 60) {
        showError('BMI must be between 10 and 60');
        return false;
    }

    // Children validation
    if (data.children < 0 || data.children > 10) {
        showError('Number of children must be between 0 and 10');
        return false;
    }

    // Check all fields are filled
    if (!data.sex || !data.smoker || !data.region) {
        showError('Please fill in all fields');
        return false;
    }

    return true;
}

/**
 * Show loading state
 */
function showLoading() {
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');

    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    submitBtn.disabled = true;
}

/**
 * Hide loading state
 */
function hideLoading() {
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');

    btnText.style.display = 'inline';
    btnLoader.style.display = 'none';
    submitBtn.disabled = false;
}

/**
 * Display prediction results
 */
function displayResults(result) {
    // Animate the amount
    animateValue(predictionAmount, 0, result.prediction, 1500);

    // Display insights
    displayInsights(result.insights);

    // Display input summary
    displayInputSummary(result.inputs);

    // Show results container with animation
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Animate number counting
 */
function animateValue(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out)
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * easeOut;

        element.textContent = current.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/**
 * Display insights
 */
function displayInsights(insights) {
    insightsList.innerHTML = '';

    insights.forEach((insight, index) => {
        const insightItem = document.createElement('div');
        insightItem.className = 'insight-item';
        insightItem.textContent = insight;
        insightItem.style.animationDelay = `${index * 0.1}s`;
        insightsList.appendChild(insightItem);
    });
}

/**
 * Display input summary
 */
function displayInputSummary(inputs) {
    inputSummary.innerHTML = '';

    const summaryData = [
        { label: 'Age', value: inputs.age },
        { label: 'Gender', value: capitalizeFirst(inputs.sex) },
        { label: 'BMI', value: inputs.bmi },
        { label: 'Children', value: inputs.children },
        { label: 'Smoker', value: capitalizeFirst(inputs.smoker) },
        { label: 'Region', value: capitalizeFirst(inputs.region) }
    ];

    summaryData.forEach(item => {
        const summaryItem = document.createElement('div');
        summaryItem.className = 'summary-item';
        summaryItem.innerHTML = `
            <div class="summary-label">${item.label}</div>
            <div class="summary-value">${item.value}</div>
        `;
        inputSummary.appendChild(summaryItem);
    });
}

/**
 * Show error message
 */
function showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.innerHTML = `
        <div class="error-content">
            <span class="error-icon">⚠️</span>
            <span class="error-message">${message}</span>
        </div>
    `;

    // Add styles
    errorDiv.style.cssText = `
        position: fixed;
        top: 100px;
        right: 24px;
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        animation: slideInRight 0.3s ease-out;
    `;

    document.body.appendChild(errorDiv);

    // Remove after 4 seconds
    setTimeout(() => {
        errorDiv.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => errorDiv.remove(), 300);
    }, 4000);
}

/**
 * Reset form and hide results
 */
function resetForm() {
    predictionForm.reset();
    resultsContainer.style.display = 'none';

    // Scroll to form
    predictionForm.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Capitalize first letter
 */
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Add smooth scrolling for navigation links
 */
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        const href = link.getAttribute('href');
        if (href.startsWith('#')) {
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        }
    });
});

/**
 * Add input animations
 */
document.querySelectorAll('.form-input').forEach(input => {
    input.addEventListener('focus', function () {
        this.parentElement.classList.add('focused');
    });

    input.addEventListener('blur', function () {
        this.parentElement.classList.remove('focused');
    });
});

/**
 * BMI Calculator Helper
 */
const bmiInput = document.getElementById('bmi');
if (bmiInput) {
    bmiInput.addEventListener('input', function () {
        const bmi = parseFloat(this.value);
        if (bmi && bmi > 0) {
            let category = '';
            let color = '';

            if (bmi < 18.5) {
                category = 'Underweight';
                color = '#4facfe';
            } else if (bmi < 25) {
                category = 'Normal';
                color = '#00f2fe';
            } else if (bmi < 30) {
                category = 'Overweight';
                color = '#f093fb';
            } else {
                category = 'Obese';
                color = '#f5576c';
            }

            // Update hint with category
            const hint = this.nextElementSibling;
            if (hint && hint.classList.contains('input-hint')) {
                hint.innerHTML = `BMI between 10-60 <span style="color: ${color}; font-weight: 600;">(${category})</span>`;
            }
        }
    });
}

/**
 * Add CSS for animations
 */
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100px);
        }
    }
    
    .error-content {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .error-icon {
        font-size: 20px;
    }
    
    .error-message {
        font-weight: 500;
    }
    
    .form-group.focused .form-label {
        color: #667eea;
    }
`;
document.head.appendChild(style);

// Log initialization
console.log('MediCost AI - Prediction System Initialized');
console.log('Version: 1.0.0');
