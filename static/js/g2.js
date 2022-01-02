var chartDom = document.getElementById('g2');
var myChart = echarts.init(chartDom);
var option;

option = {
  xAxis: {
    type: 'category',
    name: 'judgement',
    data: ['True', 'False']
  },
  yAxis: {
    type: 'value',
    name: 'amount'
  },
  series: [
    {
      data: [
        73,
        {
          value: 127,
          itemStyle: {
            color: '#a90000'
          }
        },
      ],
      type: 'bar',
      itemStyle:{
        normal:{
          label:{
            show:true,
            position:'top',
            textStyle:{
              color:'black',
              fontSize:14
            }
          }
        }
      }
    }
  ]
};

option && myChart.setOption(option);