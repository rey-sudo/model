<template>
    <div>
       <ChartMatrix3d :points="misDatos" :target="puntoCentral"/>
    </div>
</template>

<script setup>
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

</script>

<style lang="css" scoped>

</style>