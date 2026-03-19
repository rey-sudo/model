<template>
    <div  class="tab-train-center">
      <ChartMatrix3d :points="misDatos" , :target="puntoCentral" />
      <Console />
    </div>
</template>

<script setup>
const misDatos = {
  x: [
    8298.85, 3047.98, 2934.95, 5120.4, 1200.15, 7500.6, 4300.2, 9100.0, 2500.5,
    6800.3, 1500.9, 5900.4, 3100.75,
  ],
  y: [
    8759.67, 3781.44, 1110.77, 4500.3, 2100.8, 7800.9, 3200.45, 9500.1, 1800.6,
    6200.25, 2900.7, 5300.85, 4100.2,
  ],
  z: [
    62.87, 887.32, 7063.16, 1200.45, 5400.12, 300.55, 8500.9, 150.2, 4200.75,
    2100.3, 9800.6, 3300.4, 1100.95,
  ],
  labels: [
    "territory",
    "area",
    "residents",
    "sector A",
    "sector B",
    "zone alpha",
    "zone beta",
    "central",
    "outskirts",
    "urban",
    "rural",
    "industrial",
    "commercial",
  ],
};

const puntoCentral = {
  x: 4200.24,
  y: 6149.22,
  z: 2799.93,
  label: "bank",
};
</script>

<style lang="css" scoped>
.tab-train-center {
  display: grid;
  grid-template-columns: 1fr;
  grid-template-rows: 50vh 1fr;
  gap: 1rem;
}
</style>