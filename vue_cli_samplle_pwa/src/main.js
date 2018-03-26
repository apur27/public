// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import tether from 'tether'
import 'bootstrap'
import BootstrapVue from 'bootstrap-vue'
import bCarousel from 'bootstrap-vue/es/components/carousel/carousel'

global.Tether = tether
Vue.use(BootstrapVue)
Vue.component('b-carousel', bCarousel)
Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  template: '<App/>',
  components: { App }
})
