// 主页面JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // 图片预览功能
    const imageInput = document.getElementById('imageInput');
    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // 可以在这里添加图片预览功能
                console.log('选择的文件:', file.name);
            }
        });
    }
    
    // 表单提交处理
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const predictBtn = document.getElementById('predictBtn');
            if (predictBtn) {
                predictBtn.disabled = true;
                predictBtn.textContent = '识别中...';
            }
        });
    }
});