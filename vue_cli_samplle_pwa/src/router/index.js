import Vue from 'vue'
import Router from 'vue-router'
import HomeView from '@/components/HomeView'
import AboutUsView from '@/components/AboutUsView'
import ContactUsView from '@/components/ContactUsView'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Home',
      component: HomeView
    },
    {
      path: '/about',
      name: 'AboutUs',
      component: AboutUsView
    },
    {
      path: '/contact',
      name: 'ContactUs',
      component: ContactUsView
    }
  ]
})
