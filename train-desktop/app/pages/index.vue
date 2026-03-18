<template>
    <component 
      v-for="(component, name) in tabs" 
      :key="name"
      :is="component"
      v-show="tabStore.activeTab === name"
    />
</template>

<script setup lang="ts">
import { markRaw } from 'vue';
import { useTabStore } from "~/stores/tab";
import TabTrain from "@/components/TabTrain.vue";
import TabSemantic from "@/components/TabSemantic.vue";

const tabStore = useTabStore();


const tabs = {
  TabTrain: markRaw(TabTrain),
  TabSemantic: markRaw(TabSemantic)
};

</script>

<style scoped>

</style>