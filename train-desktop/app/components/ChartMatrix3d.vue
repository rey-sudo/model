<template>
  <div class="chart-wrapper">
    <div ref="plotlyChart" class="plotly-container"></div>
    <div v-if="loading" class="loading-overlay">Cargando matriz...</div>
  </div>
</template>

<script setup>
import { onMounted, ref, watch, onBeforeUnmount } from "vue";

const plotlyChart = ref(null);
const loading = ref(true);
let Plotly = null;

const props = defineProps({
  points: {
    type: Object,
    required: true,
    default: () => ({
      x: [2, -2, 2, -2],
      y: [2, 2, -2, -2],
      z: [5, 5, 5, 5],
      labels: ["Punto A", "Punto B", "Punto C", "Punto D"],
    }),
  },
  target: {
    type: Object,
    default: () => ({ x: 0, y: 0, z: 0, label: "CENTRO" }),
  },
});

const buildDataStructure = () => {
  const x = [],
    y = [],
    z = [],
    text = [],
    colors = [],
    sizes = [];

  const colorPeriferico = "#e74c3c"; 
  const colorCentro = "rgba(49, 119, 180, 1)"; 

  for (let i = 0; i < props.points.x.length; i++) {
    // 1. Punto Periférico
    x.push(props.points.x[i]);
    y.push(props.points.y[i]);
    z.push(props.points.z[i]);
    text.push(props.points.labels[i] || `Punto ${i}`);
    colors.push(colorPeriferico);
    sizes.push(8); // Tamaño normal

    // 2. Punto Central (Destino)
    x.push(props.target.x);
    y.push(props.target.y);
    z.push(props.target.z);
    text.push(props.target.label);
    colors.push(colorCentro);
    sizes.push(20); // Más grande

    // 3. Null para separar líneas
    x.push(null);
    y.push(null);
    z.push(null);
    text.push(null);
    colors.push("rgba(0,0,0,0)");
    sizes.push(0);
  }

  return { x, y, z, text, colors, sizes };
};

const drawChart = async () => {
  if (!process.client) return;

  try {
    if (!Plotly) {
      const module = await import("plotly.js-dist-min");
      Plotly = module.default || module;
    }

    const s = buildDataStructure();

    const trace = {
      type: "scatter3d",
      mode: "lines+markers",
      x: s.x,
      y: s.y,
      z: s.z,
      text: s.text,
      textposition: "top center", 
      textfont: {
        color: "#64748b", 
        size: 12
      },
      hoverinfo: "text+x+y+z",
      line: {
        color: "rgba(49, 119, 180, 1)",
        width: 2,
      },
      marker: {
        size: s.sizes, // Array de tamaños
        color: s.colors, // Array de colores
        symbol: "circle",
        line: { color: "rgba(0,0,0,1)", width: 1 },
      },
    };

    const layout = {
      autosize: true,
      height: 500,
      dragmode: "turntable", // Rotación estable
      hovermode: "closest",
      margin: { l: 0, r: 0, b: 0, t: 0 },
      scene: {
        fixedratio: true,
        aspectmode: "cube",
        xaxis: {
          title: {
            text: "X Axis",
            font: {
              color: "#64748b",
              size: 14,
              family: "Arial, sans-serif",
            },
          },
          zerolinecolor: "#000000",
          showline: true,
          mirror: false,
          showticklabels: false,
          showspikes: false
        },
        yaxis: {
          title: {
            text: "Y Axis",
            font: {
              color: "#64748b",
              size: 14,
              family: "Arial, sans-serif",
            },
          },
          zerolinecolor: "#000000",
          showline: true,
          mirror: false,
          showticklabels: false,
          showspikes: false
        },
        zaxis: {
          title: {
            text: "Z Axis",
            font: {
              color: "#64748b",
              size: 14,
              family: "Arial, sans-serif",
            },
          },
          tickfont: {
            color: "#64748b",
          },
          zerolinecolor: "#000000",
          showline: false,
          mirror: false,
          showticklabels: false,
          showspikes: false
        },
        camera: {
          projection: {
            type: "orthographic",
          },
          up: { x: 0, y: 0, z: 1 }, // Z siempre arriba
          eye: { x: 1.5, y: 1.5, z: 1.5 },
        },
      },
      margin: { l: 0, r: 0, b: 0, t: 30 },
    };

    const config = { responsive: true, displaylogo: false };

    Plotly.react(plotlyChart.value, [trace], layout, config);
    loading.value = false;
  } catch (err) {
    console.error("Error cargando Plotly:", err);
  }
};

watch(
  () => [props.points, props.target],
  () => drawChart(),
  { deep: true },
);

onMounted(() => {
  drawChart();
  window.addEventListener("resize", onResize);
});

const onResize = () => {
  if (Plotly && plotlyChart.value) Plotly.Plots.resize(plotlyChart.value);
};

onBeforeUnmount(() => {
  window.removeEventListener("resize", onResize);
});

/*

const misDatos = {
  x: [8298.85, 3047.98, 2934.95],
  y: [8759.67, 3781.44, 1110.77],
  z: [62.87, 887.32, 7063.16],
  labels: ["vehicle", "wheels", "gas"]
};

const puntoCentral = {
  x: 4200.24,
  y: 6149.22,
  z: 2799.93,
  label: 'car'
};


*/
</script>

<style scoped>
.chart-wrapper {
  width: 100%;
  height: 100%;
  overflow: hidden;
  display: flex;
  align-items: center;
  box-sizing: border-box;
  justify-content: center;
  background: var(--color-white);
  border-radius: var(--chart-radius);
  border: 1px solid var(--color-border);
}
.plotly-container {
  width: 100%;
  height: 100%;
}
.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.7);
}
</style>
