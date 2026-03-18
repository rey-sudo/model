<template>
  <div class="editor-scroll-container">
    <editor-content :editor="editor" />
  </div>
</template>

<script setup lang="ts">
import { useEditor, EditorContent } from "@tiptap/vue-3";
import StarterKit from "@tiptap/starter-kit";
import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";

const initialJson = {
  type: "doc",
  content: [
    {
      type: "paragraph",
      content: [
        {
          type: "text",
          text: `The West Bank is a landlocked territory near the Mediterranean coast of Western Asia, bordered by Jordan to the east and Israel to the north, south, and west. It is a region of immense historical and religious significance, housing ancient cities like Jericho, Hebron, and Bethlehem. Its landscape is characterized by rugged limestone hills and vast olive groves that have been cultivated for generations. Despite its cultural richness, the area is defined by a complex geopolitical reality and a fragmented administrative structure. This ongoing situation creates significant challenges for the daily lives and mobility of its residents, keeping the region at the center of international diplomatic focus.`,
        },
      ],
    },
  ],
};

const editor = useEditor({
  extensions: [StarterKit, Document, Paragraph, Text],
  content: initialJson,
  editable: false,
  editorProps: {
    attributes: {
      class: "tiptap-editor-container",
    },
  },
});
</script>

<style lang="css">
.editor-scroll-container {
  height: 100%;
  overflow-y: auto;
  padding: 1rem;
  border-radius: var(--card-radius);
  background: var(--color-white);
  border: 1px solid var(--color-border);
}

:deep(.tiptap:focus) {
  outline: none;
}

.tiptap {
  outline: none;
  min-height: 150px;
}

.tiptap :first-child {
  margin-top: 0;
}
</style>
