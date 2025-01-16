<template>
    <div class="container">
      <h2>Welcome to the Vue.js & PrimeVue App</h2>
      <Button label="Fetch Data" icon="pi pi-refresh" @click="fetchData" />
      <div v-if="loading">Loading...</div>
      <div v-if="data">
        <h3>API Response:</h3>
        <pre>{{ data }}</pre>
      </div>
    </div>
  </template>
  
  <script>
  import { ref } from 'vue'
  import Button from 'primevue/button'
  
  export default {
    name: 'Home',
    components: {
      Button
    },
    setup() {
      const data = ref(null)
      const loading = ref(false)
  
      const fetchData = async () => {
        loading.value = true
        try {
          const response = await fetch('http://127.0.0.1:5000/api/data')
          data.value = await response.json()
        } catch (error) {
          console.error('Error fetching data:', error)
        } finally {
          loading.value = false
        }
      }
  
      return { data, loading, fetchData }
    }
  }
  </script>
  
  <style scoped>
  .container {
    padding: 20px;
  }
  </style>
  