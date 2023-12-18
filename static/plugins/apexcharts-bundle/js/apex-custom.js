$(function () {
  "use strict";

  // chart 2
  var options = {
    series: [
      {
        name: "Price",
        data: [
          3400, 5100, 6000, 6300, 6400,
        ],
      },
    ],
    chart: {
      foreColor: "#9ba7b2",
      height: 360,
      type: "line",
      zoom: {
        enabled: false,
      },
      toolbar: {
        show: false,
      },
      dropShadow: {
        enabled: true,
        top: 3,
        left: 14,
        blur: 4,
        opacity: 0.1,
      },
    },
    stroke: {
      width: 5,
      curve: "smooth",
    },
    xaxis: {
      type: "month",
      categories: [
        "06-11-2023",
        "07-11-2023",
        "08-11-2023",
        "09-11-2023",
        "10-11-2023",
      ],
    },
    fill: {
      type: "gradient",
      gradient: {
        shade: "light",
        gradientToColors: ["#7d67ff"],
        shadeIntensity: 1,
        type: "horizontal",
        opacityFrom: 1,
        opacityTo: 1,
        stops: [0, 100, 100, 100],
      },
    },
    markers: {
      size: 4,
      colors: ["#7d67ff"],
      strokeColors: "#fff",
      strokeWidth: 2,
      hover: {
        size: 7,
      },
    },
    colors: ["#7d67ff"],
  };
  var chart = new ApexCharts(document.querySelector("#chartMDEC"), options);
  chart.render();

  // chart 3
  var options = {
    series: [
      {
        name: "Price",
        data: [2300, 4100, 5300, 4800, 6200],
      },
    ],
    chart: {
      foreColor: "#9ba7b2",
      height: 360,
      type: "line",
      zoom: {
        enabled: false,
      },
      toolbar: {
        show: false,
      },
      dropShadow: {
        enabled: true,
        top: 3,
        left: 14,
        blur: 4,
        opacity: 0.1,
      },
    },
    stroke: {
      width: 5,
      curve: "smooth",
    },
    xaxis: {
      type: "month",
      categories: [
        "06-11-2023",
        "07-11-2023",
        "08-11-2023",
        "09-11-2023",
        "10-11-2023",
      ],
    },
    fill: {
      type: "gradient",
      gradient: {
        shade: "light",
        gradientToColors: ["#7d67ff"],
        shadeIntensity: 1,
        type: "horizontal",
        opacityFrom: 1,
        opacityTo: 1,
        stops: [0, 100, 100, 100],
      },
    },
    markers: {
      size: 4,
      colors: ["#7d67ff"],
      strokeColors: "#fff",
      strokeWidth: 2,
      hover: {
        size: 7,
      },
    },
    colors: ["#7d67ff"],
  };
  var chart = new ApexCharts(document.querySelector("#chartBUMI"), options);
  chart.render();

  // chart ADMIN
  var options = {
    series: [
      {
        name: "Price",
        data: [6400, 5100, 6000, 6300],
      },
    ],
    chart: {
      foreColor: "#9ba7b2",
      height: 360,
      type: "line",
      zoom: {
        enabled: false,
      },
      toolbar: {
        show: false,
      },
      dropShadow: {
        enabled: true,
        top: 3,
        left: 14,
        blur: 4,
        opacity: 0.1,
      },
    },
    stroke: {
      width: 5,
      curve: "smooth",
    },
    xaxis: {
      type: "week",
      categories: ["Week 1", "Week 2", "Week 3", "Week 4"],
    },
    fill: {
      type: "gradient",
      gradient: {
        shade: "light",
        gradientToColors: ["#7d67ff"],
        shadeIntensity: 1,
        type: "horizontal",
        opacityFrom: 1,
        opacityTo: 1,
        stops: [0, 100, 100, 100],
      },
    },
    markers: {
      size: 4,
      colors: ["#7d67ff"],
      strokeColors: "#fff",
      strokeWidth: 2,
      hover: {
        size: 7,
      },
    },
    colors: ["#7d67ff"],
  };
  var chart = new ApexCharts(document.querySelector("#chartADMIN"), options);
  chart.render();
});
