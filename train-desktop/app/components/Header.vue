<template>
  <div class="header flex center gap-1">
    <UButton
      :class="{ tabActive: isTabActive('TrainTab') }"
      @click="tabStore.selectTab('TrainTab')"
      color="neutral"
      variant="ghost"
      >Train</UButton
    >

    <UButton
      :class="{ tabActive: isTabActive('Semantic') }"
      @click="tabStore.selectTab('Semantic')"
      color="neutral"
      variant="ghost"
      >Semantic</UButton
    >

    <UColorModeButton />
  </div>
</template>

<script setup lang="ts">
import { useTabStore } from "~/stores/tab";

const tabStore = useTabStore();

function isTabActive(tabName: string){
  return tabStore.activeTab === tabName
}
</script>

<style lang="css" scoped>
.header {
  border-bottom: 1px solid var(--color-border);
}

.tabActive {
  border-bottom: 1px solid var(--color-primary);
}
</style>
