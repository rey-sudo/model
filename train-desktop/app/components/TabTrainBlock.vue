<template>
  <div class="tab-train-block">
    <TabTrainBlockEditor />
    <TabTrainBlockControl />
  </div>
</template>

<script setup></script>

<style lang="css" scoped>
.tab-train-block {
  display: grid;
  grid-template-columns: 1fr;
  grid-template-rows: 40vh 1fr;
  gap: 1rem;
}
</style>
