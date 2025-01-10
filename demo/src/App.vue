<template>
  <div class="todo">
    <Tabs value="0" class="tabs">
      <TabList>
        <Tab value="0">Search Recipe</Tab>
        <Tab value="1">Classify Recipe</Tab>
      </TabList>
      <TabPanels>
        <TabPanel value="0">
          <InputText placeholder="Search..." type="text" v-model="searchNL"  class="searhcSection" />
          <br/>
          <Button label="Search" class="searhcbtn" @click="searchRecipe()"/>
        </TabPanel>
        <TabPanel value="1">
          <Card>
            <template #title> 
              Insert Your Recipe
            </template>
            <template #content>
              <Textarea placeholder="Description" rows="5" cols="30" v-model="classifyRecipe.description" class="searhcSection" />
              <h3 v-if="classifyRecipe.class != undefined"> Class: {{ classifyRecipe.class }}</h3>
              <br/>
              <Button label="Classify Recipe" class="searhcbtn" @click="classifyRecipeUser()"/>
            </template>
          </Card>
        </TabPanel>
      </TabPanels>
    </Tabs>
    <div class="container" v-if="receta.name != ''">
      <Card>
        <template #title> Class: {{ receta.cluster_id }} -  {{ receta.name }}</template>
        <template #content>
          <p class="m-0"> {{ receta.description }} </p>
        </template>
        <template #footer>
          <div class="flex gap-4 mt-1">
            <p v-if="receta.ingredients.length"> Ingredients: </p>
            <ul>
              <li v-for="(item, index) in receta.ingredients" :key="index">
                {{ item }}
              </li>
            </ul>
          </div>
        </template>
      </Card>
    </div>
    <div class="card">
      <Button icon="pi pi-list" label="Load Catalog" @click="getRecipes()"/>
      <DataTable :value="recetas" tableStyle="min-width: 50rem">
          <Column field="name" header="Name"></Column>
          <Column field="description" header="Description"></Column>
          <Column field="ingredients" header="Ingredients"></Column>
          <Column field="cluster_id" header="Class"></Column>
          <Column field="similarity" header="Similarity"></Column>
          <Column class="w-24 !text-end">
            <template #body="{ data }">
                <Button icon="pi pi-eye" @click="selectRow(data)" severity="secondary" rounded></Button>
            </template>
        </Column>
      </DataTable>
    </div>
  </div>

</template>

<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios'


const searchNL = ref("");
const recetas = ref([]);
const classifyRecipe = ref({
  name: "",
  description: "",
  class: undefined
});
const receta = ref({
  name: "",
  description: "",
  ingredients:[]
});


onMounted(() => {
	getRecipes()
});

const getRecipes = async () => {
  try {
    const response = await axios.get('http://127.0.0.1:5000/recipesList');
    recetas.value = response.data.recipes
  } catch (error) {
    console.error('Error fetching recipes:', error.message);
  }
};

const selectRow = (data) => { 
  receta.value = data
}

const searchRecipe = async () => { 
  try {
    const response = await axios.post('http://127.0.0.1:5000/query', {query: searchNL.value});
    recetas.value = response.data
  } catch (error) {
    console.error('Error fetching recipes:', error.message);
  }
}

const classifyRecipeUser = async () => { 
  try {
    const response = await axios.post('http://127.0.0.1:5000/classify', {recipe: classifyRecipe.value.description});
    classifyRecipe.value.class = response.data.class
    recetas.value = response.data.recipes
    console.log(response.data.recipes)
  } catch (error) {
    console.error('Error fetching recipes:', error.message);
  }
}



</script>

<style scoped>
.todo{
  width: 1000px;
  margin-left: 15%;
}

.tabs{
  display: flex;
  justify-content: center; 
  align-items: center;    
  width: 100%;            
  height: 300px;         
  box-sizing: border-box; 
}

.container {
  display: flex;
  justify-content: center; 
  align-items: center;  
  width: 80%;           
  margin-left: 10%;
  height: 400px;         
  box-sizing: border-box; 
  margin-bottom:30px;
}

.searhcSection{
  margin-top: 20px;
  width: 700px;
}

.searhcbtn{
  width: 50%;
  margin-left: 25%;
  margin-top:15px;

}

.Card {
  max-width: 500px;      
  width: 90%;             
  margin: auto;          
  padding: 20px;        
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
  border-radius: 8px;     
  background-color: white; 
}


</style>
