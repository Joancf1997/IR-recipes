<template>
  <div class="todo">
    <div id="title"> Recipe System  </div>
    <div id="wrapper">
      <div class="actionsSpace"> 
        <button id="clearbtn" @click="clearCanvas()"> Clear </button>
      </div>
    </div>
  </div>
</template>

<style scoped>

.todo{
  width: 100vw;
}

#title{ 
  width: 100%;
  height: 100px;
  font-weight: bold;
  font-size: 4em;
  margin-top: 50px;
  text-align: center;
}

#clearbtn{ 
  width: 30%;
  float: left;
  border: 2px solid rgba(255, 1, 1, 0.593);
  color: red;
  background-color: transparent;
  height: 100%;
  border-radius: 20px;
  font-size: 2em;
  font-weight: 300;
}

</style>



<script setup>
import api from "@/boot/axios"
import { ref } from 'vue';




</script>
