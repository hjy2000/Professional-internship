var chartDom = document.getElementById('g3');
var myChart = echarts.init(chartDom);
var option;

option = {
  xAxis: {
    name: 'elements',
  },
  yAxis: {
    name: 'judgement',
    min:0,
    max:1,
    axisLabel:{
        formatter: function (value) {
            var texts = [];
            if(value==0){
                texts.push('False');
            }
            else if (value == 1) {
                texts.push('True');
            }
            return texts;
        }
    }
  },
  series: [
    {
      symbolSize: 20,
      data: [
        [1, 1],
        [2, 1],
        [3, 0],
        [4, 0],
        [5, 0],
      ],
      type: 'scatter'
    }
  ]
};

option && myChart.setOption(option);