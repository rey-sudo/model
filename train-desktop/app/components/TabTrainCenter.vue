<template>
    <div  class="tab-train-center">
      <ChartMatrix3d :points="misDatos" , :target="puntoCentral" />
      <Console />
    </div>
</template>

<script setup>
const misDatos = {
  x: [
-0.9064540150357536, 
  ],
  y: [
-0.8651114971994216,
  ],
  z: [
-0.24386952799834993
  ],
  labels: [
    "territory"
  ],
};

const puntoCentral = {
  x: -0.20715636462139808,
  y: -0.10693623942950192,
  z: 0.21630182937167164,
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