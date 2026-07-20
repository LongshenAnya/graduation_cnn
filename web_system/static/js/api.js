// API调用相关函数
function submitFeedback(correction) {
    const predictionId = window.currentPredictionId;
    if (!predictionId) {
        alert('没有可反馈的预测结果');
        return;
    }
    
    fetch('/api/feedback/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify({
            prediction_id: predictionId,
            correction: correction
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage('感谢您的反馈！', 'success');
        } else {
            showMessage('反馈提交失败: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        showMessage('网络错误，请重试', 'danger');
    });
}

function getCSRFToken() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]');
    return token ? token.value : '';
}

function showMessage(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}