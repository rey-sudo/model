<template>
  <component :is="tabs[tabStore.activeTab as TabKey]" class="mt-4" />
</template>

<script setup lang="ts">
import TrainTab from "@/components/TabTrain.vue";
import { useTabStore } from "~/stores/tab";

const tabStore = useTabStore();

const tabs = {
  TrainTab: TrainTab,
};

type TabKey = keyof typeof tabs
</script>

<style lang="css" scoped></style>
