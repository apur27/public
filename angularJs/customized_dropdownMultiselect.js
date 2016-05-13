(function () {
    'use strict';

    angular
        .module('registrationModule')
        .directive('dropdownMultiselect', [dropdownMultiselect]);

    
    function dropdownMultiselect() {
   return {
       restrict: 'E',
       scope:{           
            model: '=',
            options: '=',
            pre_selected: '=preSelected'
       },
       template: "<span class='btn-group' data-ng-class='{open: open}' style=\"width:200px\">"+
        "<button class='btn btn-small'><span ng-show=\"!model.length\">  ----------------------------- </span><span ng-repeat='sel in model' ng-show=\"model.length\"> {{sel}} </span></button>" +
                "<button class='btn btn-small dropdown-toggle' data-ng-click='open=!open;openDropdown()'><span class='caret'></span></button>"+
                "<ul class='dropdown-menu' aria-labelledby='dropdownMenu'>" + 
                    "<li><a data-ng-click='selectAll()'><i class='glyphicon glyphicon-ok'>" +
                    "</i>  Check All</a></li>" +
                    "<li><a data-ng-click='deselectAll();'><i class='glyphicon glyphicon-remove'></i>  Uncheck All</a></li>" +
                    "<li class='divider'></li>" +
                    "<li data-ng-repeat='option in options'><a data-ng-click='setSelectedItem()'><i data-ng-class='isChecked(option.id)'></i>  {{option.name}}<span/></a></li>" +
                "</ul>" +                
            "</span>" ,
       controller: function($scope){
           
           $scope.openDropdown = function(){        
                                                        
            };
           
            $scope.selectAll = function () {
                $scope.model = _.pluck($scope.options, 'id');
                
            };            
            $scope.deselectAll = function() {
                $scope.model=[];
                
            };
            $scope.setSelectedItem = function(){
                var id = this.option.id;
                if (_.contains($scope.model, id)) {
                    $scope.model = _.without($scope.model, id);
                } else {
                    $scope.model.push(id);
                }
                
                return false;
            };
            $scope.isChecked = function (id) {                 
                if (_.contains($scope.model, id)) {
                    return 'glyphicon glyphicon-check';
                } else {
                    return 'glyphicon glyphicon-unchecked';
                }
                
            };                                 
       }
   } 
}
})();