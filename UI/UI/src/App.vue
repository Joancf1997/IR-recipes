
<template>
  <div>
    <FloatLabel>
      <InputText id="username" v-model="value" />
      <label for="username">Username</label>
    </FloatLabel>
  
  
    <DataTable :value="products" tableStyle="min-width: 50rem">
        <Column field="code" header="Code"></Column>
        <Column field="name" header="Name"></Column>
        <Column field="category" header="Category"></Column>
        <Column field="quantity" header="Quantity"></Column>
    </DataTable>
  </div>
</template>

<script setup>
  import {ref} from 'vue'
  import FloatLabel from 'primevue/floatlabel';
  import InputText from 'primevue/inputtext';

  import DataTable from 'primevue/datatable';
  import Column from 'primevue/column';

  const products = ref();


  const value = ref("")

</script>


<style scoped>

</style>
