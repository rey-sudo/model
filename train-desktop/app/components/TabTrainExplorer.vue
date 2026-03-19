<script setup lang="ts">
import type { TreeItem } from "@nuxt/ui";

const items = ref<TreeItem[]>([
  {
    label: "dicts/",
    defaultExpanded: true,
    children: [
      {
        label: "unicode.json",
        icon: "lucide-file-braces",
      },
    ],
  },
  {
    label: "blocks/",
    defaultExpanded: true,
    children: [
      {
        label: "b_0_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_1_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_2_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_3_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_4_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_5_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_6_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_7_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_8_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_9_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_10_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_11_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_12_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_13_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_14_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_15_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_16_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_17_100.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_18_100.json",
        icon: "lucide-file-braces",
      },
    ],
  },
]);
</script>

<template>
  <UTree :items="items" />
</template>
