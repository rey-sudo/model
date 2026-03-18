import { defineStore } from 'pinia'
import { ref, computed } from "vue";

export const useTabStore = defineStore("counter", () => {
  const activeTab = ref("TabTrain");

  function selectTab(tabName: string) {
    activeTab.value = tabName;
  }
  return {
    activeTab,
    selectTab
  };
});
