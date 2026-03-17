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

  const colorPeriferico = "#e74c3c"; // Rojo
  const colorCentro = "#3177b4"; // Azul

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
    sizes.push(18); // Más grande

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
      mode: "lines+markers+text",
      x: s.x,
      y: s.y,
      z: s.z,
      text: s.text,
      hoverinfo: "text+x+y+z",
      line: {
        color: "#3177b4",
        width: 2,
        opacity: 0.4,
      },
      marker: {
        size: s.sizes, // Array de tamaños
        color: s.colors, // Array de colores
        symbol: "circle",
        line: { color: "white", width: 1 },
      },
    };

    const layout = {
      autosize: true,
      height: 600,
      dragmode: "turntable", // Rotación estable
      scene: {
        fixedratio: true,
        aspectmode: "cube",
        xaxis: {
          title: "X",
          zerolinecolor: "#000000",
          showline: true,

          showbackground: true,
          backgroundcolor: "#f0f0f0",
        },
        yaxis: {
          title: "Y",
          zerolinecolor: "#000000",
          showline: true,
        },
        zaxis: {
          title: "Z",
          zerolinecolor: "#000000",
          showline: true,
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
</script>

<style scoped>
.chart-wrapper {
  position: relative;
  width: 100%;
  background: #fff;
  border-radius: 8px;
}
.plotly-container {
  width: 100%;
  min-height: 600px;
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
