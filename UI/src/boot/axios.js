import axios from 'axios';


const api = axios.create({
    baseURL: 'http://model-api:3000'
})

export default api;
