<template>
  <div class="mdl-layout mdl-js-layout mdl-layout--fixed-header ">
    <header class="mdl-layout__header">
      <div class="mdl-layout__header-row">
        <span class="mdl-layout-title">Sample VUE PWA</span>
      </div>
    </header>
    <div class="mdl-layout__drawer">
      <span class="mdl-layout-title">Sample VUE PWA</span>
      <nav class="mdl-navigation">
        <router-link class="mdl-navigation__link" to="/" @click.native="hideMenu">Home</router-link>
        <router-link class="mdl-navigation__link" to="/about" @click.native="hideMenu">About us</router-link>
        <router-link class="mdl-navigation__link" to="/contact" @click.native="hideMenu">Contact us</router-link>
      </nav>
    </div>
    <main class="mdl-layout__content">
      <div class="page-content">
        <router-view></router-view>
      </div>
    </main>
  </div>
</template>

<script>
require('material-design-lite')
export default {
  name: 'app'
}
</script>

<style>

  @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
  @import url('https://cdnjs.cloudflare.com/ajax/libs/material-design-lite/1.3.0/material.light_green-blue.min.css');
  @import url('https://npmcdn.com/tether@1.2.4/dist/js/tether.min.js');
  @import'~bootstrap/dist/css/bootstrap.css';
  @import'~bootstrap-vue/dist/bootstrap-vue.css';
</style>
