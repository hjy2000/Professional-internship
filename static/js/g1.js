var chartDom = document.getElementById('g1');
var myChart = echarts.init(chartDom);
var option;

option = {
    title : {
            show:true,
            subtext: '(元素名字过长不予显示，详见文档）',
            x: 'center',
            y: 'top',
            padding: 10
    },
    xAxis: {
        show: false,
        type: 'category',
        name: 'category',
        nameLocation: 'start',
        axisLabel: {
              interval: 0,
              formatter: function(value) {
                  return value.split("").join("\n");
              }
        },
        data: ['location','industry','title','description_wordsLen','requirements_wordsLen','department','company_profile_wordsLen','function','benefits_wordsLen','description_tfidf46',
        'requirements_tfidf8','salary_range_start','description_tfidf5','required_experience','company_profile_tfidf7','description_tfidf13','required_education','description_tfidf18','requirements_tfidf16','benefits_tfidf1',
        'title_wordsLen','company_profile_tfidf16','requirements_tfidf13','description_tfidf17','description_tfidf42','description_tfidf44','description_tfidf47','description_tfidf25','description_tfidf36','description_tfidf23',
        'requirements_tfidf19','requirements_tfidf2','description_tfidf0','company_profile_tfidf9','salary_range_end',"employment_type","has_company_logo","description_tfidf6","description_tfidf22","benefits_tfidf2",
        'description_tfidf37',"description_tfidf38","description_tfidf8","description_tfidf1","description_tfidf19","description_tfidf24","description_tfidf32","company_profile_tfidf2","description_tfidf2","description_tfidf3",
        "requirements_tfidf17","description_tfidf35","description_tfidf7","description_tfidf11","company_profile_tfidf12","description_tfidf31","company_profile_tfidf3","company_profile_tfidf5","description_tfidf33","requirements_tfidf0",
        "requirements_tfidf11","company_profile_tfidf4","description_tfidf34","requirements_tfidf7","description_tfidf41","description_tfidf16","company_profile_tfidf17","requirements_tfidf9","description_tfidf43","has_questions",
        "description_tfidf27","description_tfidf4","description_tfidf9","company_profile_tfidf15","description_tfidf12","requirements_tfidf14","description_tfidf26",'company_profile_tfidf14',"benefits_tfidf4","description_tfidf15",
        "description_tfidf39","requirements_tfidf10","requirements_tfidf5","description_tfidf29","requirements_tfidf12","requirements_tfidf18","benefits_tfidf7","requirements_tfidf4","benefits_tfidf11","description_tfidf14",
        "description_tfidf20","description_tfidf21","company_profile_tfidf18","benefits_tfidf10","requirements_tfidf15","benefits_tfidf8","requirements_tfidf6","description_tfidf10","company_profile_tfidf13","company_profile_tfidf1",
        "company_profile_tfidf0","benefits_tfidf3","company_profile_tfidf20","company_profile_tfidf23","company_profile_tfidf21","description_tfidf30","description_tfidf40","requirements_tfidf1","requirements_tfidf3","description_tfidf45",
        "benefits_tfidf0","company_profile_tfidf6","benefits_tfidf6","company_profile_tfidf19",'description_tfidf28',"company_profile_tfidf11","benefits_tfidf5","company_profile_tfidf10","benefits_tfidf9","company_profile_tfidf22",
        "telecommuting","company_profile_tfidf8"]
      },
  yAxis: {
    type: 'value',
    name: 'value'
  },
  series: [{
      data: ['383.6','348.4','276.8','242.4','171.0','151.8','151.6','143.4','110.4','103.8',
      '88.8','83.0','81.8','81.4','75.8','74.4','71.0','70.6','70.2','68.0',
      '65.0','64.8','62.2','61.6','61.2','60.6','59.0','56.4','56.0','54.4',
      '54.0','53.0','50.4','48.8','46.6','44.2','43.0','42.6','39.6','39.6',
      '36.4','35.8','35.4','35.4','34.8','34.2','34.0','33.6','33.0','32.8',
      '32.0','31.6','31.2','30.8','29.6','29.0','27.6','26.6','25.4','25.0',
      '24.6','24.4','23.8','23.6','23.6','23.6','23.4','23.4','23.0','22.6',
      '22.4','22.4','22.2','22.2','21.8','21.8','21.0','20.8','20.8','20.6',
      '20.4','18.8','18.4','18.4','18.0','17.6','17.4','17.4','17.2','17.0',
      '16.8','16.4','16.4','16.4','14.8','13.8','13.2','13.0','12.8','12.8',
      '12.8','11.4','10.6','9.6','9.2','9.0','8.6','8.4','8.0','8.0','7.6',
      '7.4','6.8','6.6','6.6','6.6','5.8','4.2','3.8','3.4','3.2','2.6','2.4'],
      type: 'bar'
  }],
  grid: {
    top: '15%',
    left: '10%',
    right: '10%'
  }
};

option && myChart.setOption(option);