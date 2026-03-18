<template>
  <div class="container">
    <div class="tabs">
      <UButton
        v-for="tab in tabs"
        :key="tab"
        :class="{ tabActive: isTabActive(tab) }"
        @click="selectTab(tab)"
        color="neutral"
        :variant="getButtonVariant(tab)"
        size="sm"
        >{{ tab }}</UButton
      >
    </div>
    <div>content</div>
  </div>
</template>

<script setup lang="ts">
const tabs = [
  "NOUN",
  "PROPN",
  "VERB",
  "ADJ",
  "PRON",
  "ADV",
  "ADP",
  "DET",
  "AUX",
];
const activeTab = ref("NOUN");

function isTabActive(tabName: string) {
  return activeTab.value === tabName;
}

function selectTab(tabName: string) {
  return (activeTab.value = tabName);
}

function getButtonVariant(tabName: string) {
  return isTabActive(tabName) ? "soft" : "ghost";
}
</script>

<style lang="css" scoped>
.container {
  height: 100%;
  overflow: hidden;
  border-radius: var(--card-radius);
  background: var(--color-white);
  border: 1px solid var(--color-border);
}

.tabs{
  border-bottom: 1px solid var(--color-border);
}

.tabActive {
  border-bottom: 1px solid var(--color-primary);
}
</style>
