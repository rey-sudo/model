<template>
  <div class="container">
    <div class="tabs">
      <UButton
        v-for="tab in tabs"
        :key="tab"
        class="tab"
        :class="{ tabActive: isTabActive(tab) }"
        @click="selectTab(tab)"
        color="neutral"
        :variant="getButtonVariant(tab)"
        size="md"
        >{{ tab }}</UButton
      >
    </div>
    <div>
      <UTable :data="data" class="flex-1" />
    </div>
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

const data = ref([
  {
    sign: "bank",
    index: 132,
    hash: "asxzdg"
  },
  { sign: "territory", index: 276, hash: "gdfgx" },
   { sign: "area", index: 226, hash: "xsda" },
]);

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
  padding: 1rem;
  overflow: hidden;
  caret-color: transparent;
  border-radius: var(--card-radius);
  background: var(--color-white);
  border: 1px solid var(--color-border);
}

.tab {
  font-weight: 400;
  color: var(--color-text-secondary);
}

.tabs {
  border-bottom: 1px solid var(--color-border);
}

.tabActive {
  color: var(--color-text);
  border-bottom: 1px solid var(--color-primary);
}
</style>
