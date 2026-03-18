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
          text: `As part of an interdisciplinary field, psychologists are involved in research on perception, cognition, attention, emotion, intelligence, subjective experiences, motivation, brain functioning, and personality. Psychologists' interests extend to interpersonal relationships, psychological resilience, family resilience, and other areas within social psychology. They also consider the unconscious mind. Research psychologists employ empirical methods to infer causal and correlational relationships between psychosocial variables. Some, but not all, clinical and counseling psychologists rely on symbolic interpretation.`,
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
  /* Define la altura que prefieras */
  height: 500px; 
  /* O usa max-height si quieres que crezca hasta cierto punto */
  /* max-height: 70vh; */
  
  overflow-y: auto;
  padding: 1rem;
  border-radius: 8px;
}

/* Opcional: Quitar el outline azul al hacer foco para que se vea más limpio */
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
