<template>
  <div class="chart-wrapper">
    <div ref="plotlyChart" class="plotly-container"></div>

    <div v-if="loading" class="loading-overlay">
      Cargando visualización 3D...
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref, watch, onBeforeUnmount } from "vue";

const plotlyChart = ref(null);
const loading = ref(true);
let Plotly = null;

const props = defineProps({
  // Puntos periféricos
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
  // Punto al que todos apuntan (por defecto el origen)
  target: {
    type: Object,
    default: () => ({ x: 0, y: 0, z: 0, label: "CENTRO" }),
  },
});

const buildStarStructure = () => {
  const x = [];
  const y = [];
  const z = [];
  const text = [];

  // Construimos segmentos individuales: Punto[i] -> Target -> null
  for (let i = 0; i < props.points.x.length; i++) {
    // 1. Punto de la matriz
    x.push(props.points.x[i]);
    y.push(props.points.y[i]);
    z.push(props.points.z[i]);
    text.push(props.points.labels[i] || `Punto ${i}`);

    // 2. Punto destino (Centro)
    x.push(props.target.x);
    y.push(props.target.y);
    z.push(props.target.z);
    text.push(props.target.label);

    // 3. Corte de línea (Null) para que no se unan entre sí
    x.push(null);
    y.push(null);
    z.push(null);
    text.push(null);
  }

  return { x, y, z, text };
};

const drawChart = async () => {
  if (!process.client) return;

  // Carga dinámica de Plotly para evitar errores de SSR
  if (!Plotly) {
    const module = await import("plotly.js-dist-min");
    Plotly = module.default || module;
  }

  const starData = buildStarStructure();

  const trace = {
    type: "scatter3d",
    mode: "lines+markers",
    x: starData.x,
    y: starData.y,
    z: starData.z,
    text: starData.text,
    hoverinfo: "text+x+y+z",
    line: {
      color: "#3177b4",
      width: 3,
      opacity: 0.6,
    },
    marker: {
      size: 8,
      color: "#e74c3c",
      symbol: "circle",
      line: {
        color: "white",
        width: 1,
      },
    },
  };

  const layout = {
    autosize: true,
    height: 600,
    dragmode: "turntable",
    scene: {
      xaxis: { title: "X", backgroundcolor: "#f0f0f0", showbackground: true },
      yaxis: { title: "Y", backgroundcolor: "#f0f0f0", showbackground: true },
      zaxis: { title: "Z", backgroundcolor: "#f0f0f0", showbackground: true },
      camera: {
        up: { x: 0, y: 0, z: 1 },
        center: { x: 0, y: 0, z: 0 },
        eye: { x: 1.5, y: 1.5, z: 1.2 },
      },
      aspectmode: "cube",
    },
    margin: { l: 0, r: 0, b: 0, t: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
  };

  const config = {
    scrollZoom: true,
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"],
  };

  Plotly.react(plotlyChart.value, [trace], layout, config);
  loading.value = false;
};

// Observar cambios en los datos para redibujar
watch(
  () => [props.points, props.target],
  () => {
    drawChart();
  },
  { deep: true },
);

onMounted(() => {
  drawChart();
  window.addEventListener("resize", () =>
    Plotly?.Plots.resize(plotlyChart.value),
  );
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", () =>
    Plotly?.Plots.resize(plotlyChart.value),
  );
});
</script>

<style scoped>
.chart-wrapper {
  position: relative;
  width: 100%;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
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
  background: rgba(255, 255, 255, 0.8);
  font-family: sans-serif;
  z-index: 10;
}
</style>
