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
        label: "b_0.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_1.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_2.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_3.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_4.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_5.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_6.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_7.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_8.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_9.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_10.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_11.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_12.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_13.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_14.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_15.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_16.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_17.json",
        icon: "lucide-file-braces",
      },
      {
        label: "b_18.json",
        icon: "lucide-file-braces",
      },
    ],
  },
]);
</script>

<template>
  <UTree :items="items" />
</template>
