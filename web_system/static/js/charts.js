// 图表相关JavaScript
let staticCharts = {};
let dynamicCharts = {};

function initCharts() {
    // 初始化静态图表
    staticCharts.accuracy = echarts.init(document.getElementById('accuracyChart'));
    staticCharts.time = echarts.init(document.getElementById('timeChart'));
    staticCharts.size = echarts.init(document.getElementById('sizeChart'));
    
    // 初始化动态图表
    dynamicCharts.confidence = echarts.init(document.getElementById('confidenceChart'));
    dynamicCharts.modelTime = echarts.init(document.getElementById('modelTimeChart'));
    dynamicCharts.ratio = echarts.init(document.getElementById('ratioChart'));
    
    loadStaticData();
    loadDynamicData();
    
    // 每30秒更新动态数据
    setInterval(loadDynamicData, 30000);
}

function loadStaticData() {
    fetch('/charts/api/stats/static/')
        .then(response => response.json())
        .then(data => {
            updateStaticCharts(data);
        })
        .catch(error => console.error('加载静态数据失败:', error));
}

function loadDynamicData() {
    fetch('/api/stats/dynamic/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        updateDynamicCharts(data);
    })
    .catch(error => console.error('加载动态数据失败:', error));
}

function updateStaticCharts(data) {
    // 准确率图表
    staticCharts.accuracy.setOption({
        title: { text: '模型准确率对比' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: data.map(d => d.model) },
        yAxis: { type: 'value', name: '准确率 (%)' },
        series: [{ type: 'bar', data: data.map(d => d.accuracy) }]
    });
    
    // 推理时间图表
    staticCharts.time.setOption({
        title: { text: '模型推理时间对比' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: data.map(d => d.model) },
        yAxis: { type: 'value', name: '时间 (ms)' },
        series: [{ type: 'bar', data: data.map(d => d.time) }]
    });
    
    // 模型大小图表
    staticCharts.size.setOption({
        title: { text: '模型大小对比' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: data.map(d => d.model) },
        yAxis: { type: 'value', name: '大小 (MB)' },
        series: [{ type: 'bar', data: data.map(d => d.size) }]
    });
}

function updateDynamicCharts(data) {
    // 更新今日识别次数
    const todayCountEl = document.getElementById('todayCount');
    if (todayCountEl) {
        todayCountEl.textContent = data.today_count;
    }
    
    // 置信度趋势图
    dynamicCharts.confidence.setOption({
        title: { text: '最近10次识别置信度趋势' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: Array.from({length: data.recent_confidences.length}, (_, i) => `第${i+1}次`) },
        yAxis: { type: 'value', name: '置信度 (%)' },
        series: [{ 
            type: 'line', 
            data: data.recent_confidences.map(c => (c * 100).toFixed(2)),
            smooth: true
        }]
    });
    
    // 模型推理时间图
    dynamicCharts.modelTime.setOption({
        title: { text: '模型平均推理耗时' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: ['ResNet', 'VGG', 'MobileNet'] },
        yAxis: { type: 'value', name: '时间 (ms)' },
        series: [{ type: 'bar', data: ['resnet', 'vgg', 'mobilenet'].map(m => data.model_times[m] || 0) }]
    });
    
    // 猫狗比例饼图
    const ratioData = [];
    if (data.cat_dog_ratio.cat) ratioData.push({ name: '猫', value: data.cat_dog_ratio.cat });
    if (data.cat_dog_ratio.dog) ratioData.push({ name: '狗', value: data.cat_dog_ratio.dog });
    
    dynamicCharts.ratio.setOption({
        title: { text: '猫狗识别比例' },
        tooltip: { trigger: 'item', formatter: '{a} <br/>{b}: {c} ({d}%)' },
        series: [{
            name: '识别结果',
            type: 'pie',
            radius: '50%',
            data: ratioData,
            emphasis: { 
                itemStyle: { 
                    shadowBlur: 10, 
                    shadowOffsetX: 0, 
                    shadowColor: 'rgba(0, 0, 0, 0.5)' 
                } 
            }
        }]
    });
}

// 窗口大小改变时调整图表大小
window.addEventListener('resize', function() {
    Object.values(staticCharts).forEach(chart => chart.resize());
    Object.values(dynamicCharts).forEach(chart => chart.resize());
});