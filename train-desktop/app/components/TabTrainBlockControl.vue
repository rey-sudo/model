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
    <USeparator />
    <div>
      <UTable :data="data" :columns="columns" class="flex-1" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { h, resolveComponent } from "vue";
import type { TableColumn, TableRow } from "@nuxt/ui";

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

const UBadge = resolveComponent("UBadge");

type Payment = {
  id: number;
  sign: string;
  status: string;
};

const data = ref<Payment[]>([
  { id: 0, sign: "bank", status: "yes" },
  { id: 1, sign: "territory", status: "yes" },
  { id: 2, sign: "area", status: "not" },
]);

const columns: TableColumn<Payment>[] = [
  {
    accessorKey: "id",
    header: "#",
    cell: ({ row }) => `${row.getValue("id")}`,
  },
  {
    accessorKey: "sign",
    header: "Sign",
    cell: ({ row }) => {
      return row.getValue("sign")
    },
  },
  {
    accessorKey: "status",
    header: "Indexed",
    meta: {
      class: {
        th: "text-right",
        td: "text-right font-medium",
      },
    },

    footer: ({ column }) => {
      const total = column
        .getFacetedRowModel()
        .rows.reduce(
          (acc: number, row: TableRow<Payment>) =>
            acc + Number.parseFloat(row.getValue("id")),
          0,
        );

      return `Total: 2 / ${total}`;
    },

    cell: ({ row }) => {
      const color = {
        yes: "success" as const,
        not: "error" as const,
        refunded: "neutral" as const,
      }[row.getValue("status") as string];

      return h(UBadge, { class: "capitalize", variant: "subtle", color }, () =>
        row.getValue("status"),
      );
    },
  },
];

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

.tabActive {
  color: var(--color-text);
  border-bottom: 1px solid var(--color-primary);
}
</style>
