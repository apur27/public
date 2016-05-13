(function () {
    'use strict';

    angular
        .module('registrationModule')
        .controller('disabilitySelectionController', ['$scope', '$rootScope', '$modalInstance', 'disabilityStudent', 'studentData', 'studentDisability', 'urlsHelper', 'authService', disabilitySelectionController]);

    function disabilitySelectionController($scope, $rootScope, $modalInstance, disabilityStudent, studentData, studentDisability, urlsHelper, authService) {
        $scope.categories = [
                        { "name": "Physical", "id": "Physical" },
                        { "name": "Cognitive", "id": "Cognative" },
                        { "name": "Sensory", "id": "Sensory" },
                        { "name": "Social/Emotional", "id": "Social" }

        ];
        $scope.DATypesBraille = [{ "name": "Braille format", "id": "BrailleFormat", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true }];
        $scope.DATypesElectronic = [{ "name": "Electronic test format", "id": "ElectronicFormat", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true }];
        $scope.DATypesBlackAndWhite = [{ "name": "Black and white print format", "id": "BWFormat", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true }];
        $scope.DATypes = [
                            { "name": "Extra time", "id": "ExtraTime", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true },
                            { "name": "Rest breaks", "id": "RestBreaks", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true },
                            { "name": "Oral/Sign support", "id": "OralSupport", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true },
                            { "name": "Scribe", "id": "Scribe", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true },
                            { "name": "Support person", "id": "SupportPerson", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true }];

        $scope.DASubTypeLargePrintFormats = [
                                            { "name": "A4, N18 font", "id": "N18-A4", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true },
                                            { "name": "A4, N24 font", "id": "N24-A4", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true },
                                            { "name": "A3, N18 font", "id": "N18-A3", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true },
                                            { "name": "A3, N24 font", "id": "N24-A3", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true },
                                            { "name": "A3, N36 font", "id": "N36-A3", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true }];

        $scope.DASubTypeAssistiveTechnologies = [{ "name": "Screen reader or other assistive technology", "id": "ScreenReader", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true },
                                               { "name": "Use of a computer for the writing task (no spelling or grammar check)", "id": "UseOfComputer", "Lang": true, "Writ": true, "Read": true, "Nume": true, "NumeCalc": true, "NumeNonCalc": true }];

        $scope.student = {
            "Url": "",
            "SchoolCode": "",
            "FirstName": "",
            "LastName": "",
            "YearLevel": null,
            "Gender": "",
            "DOB": "",
            "IndigenousStatus": "",
            "StudentId": "",
            "HasDisability": false
        };

        $scope.tests = {
            "lang": "1" ,
            "writ": "2" ,
            "read": "3",
            "nume": "4",
            "numeCalc": "29",
            "numeNonCalc": "30"
        };


        $scope.disabilitySelections = {
            category: {
                PRAStudentID: "",
                Physical: "",
                Social: "",
                Cognative: "",
                Sensory: ""
            },
            lang: {
                PRAStudentID: "",
                TestID: "",
                DACodes: "",                
                AssistiveTechCode: null,
                LargePrintCode: null
            },
            langDACodesArray: [],
            writ: {
                PRAStudentID: "",
                TestID: "",
                DACodes: "",
                AssistiveTechCode: null,
                LargePrintCode: null
            },
            writDACodesArray: [],
            read: {
                PRAStudentID: "",
                TestID: "",
                DACodes: "",
                AssistiveTechCode: null,
                LargePrintCode: null
            },
            readDACodesArray: [],
            nume: {
                PRAStudentID: "",
                TestID: "",
                DACodes: "",
                AssistiveTechCode: null,
                LargePrintCode: null
            },
            numeDACodesArray: [],
            numeCalc: {
                PRAStudentID: "",
                TestID: "",
                DACodes: "",
                AssistiveTechCode: null,
                LargePrintCode: null
            },
            numeCalcDACodesArray: [],
            numeNonCalc: {
                PRAStudentID: "",
                TestID: "",
                DACodes: "",
                AssistiveTechCode: null,
                LargePrintCode: null
            },
            numeNonCalcDACodesArray: []

        };
        
        
        _.assign($scope.student, disabilityStudent);
        
         
        studentDisability.getStudentDisabilityCategoryByUrl($scope.student.Url).$promise.then(function (result) {
            $scope.disabilitySelections.category = result;
        }, function (error) {
            $scope.disabilitySelections.category.PRAStudentID = urlsHelper.getLastID($scope.student.Url);
        });
        
        $scope.noLanguage = false;
        studentDisability.getStudentDisabilityTypeByUrl($scope.student.Url, $scope.tests.lang).$promise.then(function (result) {
            $scope.disabilitySelections.lang = result;
            if ($scope.disabilitySelections.lang.DACodes != undefined) {
                $scope.disabilitySelections.langDACodesArray = $scope.disabilitySelections.lang.DACodes.split(',');                
            }
                
        }, function (error) {
            $scope.disabilitySelections.lang.PRAStudentID = urlsHelper.getLastID($scope.student.Url).toString();
            $scope.disabilitySelections.lang.TestID = $scope.tests.lang;
            $scope.noLanguage = true;
        });
        
        $scope.noWriting = false;
        studentDisability.getStudentDisabilityTypeByUrl($scope.student.Url, $scope.tests.writ).$promise.then(function (result) {
            $scope.disabilitySelections.writ = result;
            if ($scope.disabilitySelections.writ.DACodes != undefined) {
                $scope.disabilitySelections.writDACodesArray = $scope.disabilitySelections.writ.DACodes.split(',');
            }

        }, function (error) {
            $scope.disabilitySelections.writ.PRAStudentID = urlsHelper.getLastID($scope.student.Url).toString();
            $scope.disabilitySelections.writ.TestID = $scope.tests.writ;
            $scope.noWriting = true;
        });
        $scope.noReading = false;
        studentDisability.getStudentDisabilityTypeByUrl($scope.student.Url, $scope.tests.read).$promise.then(function (result) {
            $scope.disabilitySelections.read = result;
            if ($scope.disabilitySelections.read.DACodes != undefined) {
                $scope.disabilitySelections.readDACodesArray = $scope.disabilitySelections.read.DACodes.split(',');
            }

        }, function (error) {
            $scope.disabilitySelections.read.PRAStudentID = urlsHelper.getLastID($scope.student.Url).toString();
            $scope.disabilitySelections.read.TestID = $scope.tests.read;
            $scope.noReading = true;
        });
        $scope.noNumeracy = false;
        studentDisability.getStudentDisabilityTypeByUrl($scope.student.Url, $scope.tests.nume).$promise.then(function (result) {
            $scope.disabilitySelections.nume = result;
            if ($scope.disabilitySelections.nume.DACodes != undefined) {
                $scope.disabilitySelections.numeDACodesArray = $scope.disabilitySelections.nume.DACodes.split(',');
            }

        }, function (error) {
            $scope.disabilitySelections.nume.PRAStudentID = urlsHelper.getLastID($scope.student.Url).toString();
            $scope.disabilitySelections.nume.TestID = $scope.tests.nume;
            $scope.noNumeracy = true;
        });
        $scope.noNumeracyCalc = false;
        studentDisability.getStudentDisabilityTypeByUrl($scope.student.Url, $scope.tests.numeCalc).$promise.then(function (result) {
            $scope.disabilitySelections.numeCalc = result;
            if ($scope.disabilitySelections.numeCalc.DACodes != undefined) {
                $scope.disabilitySelections.numeCalcDACodesArray = $scope.disabilitySelections.numeCalc.DACodes.split(',');
            }

        }, function (error) {
            $scope.disabilitySelections.numeCalc.PRAStudentID = urlsHelper.getLastID($scope.student.Url).toString();
            $scope.disabilitySelections.numeCalc.TestID = $scope.tests.numeCalc;
            $scope.noNumeracyCalc = true;
        });
        $scope.noNumeracyNonCalc = false;
        studentDisability.getStudentDisabilityTypeByUrl($scope.student.Url, $scope.tests.numeNonCalc).$promise.then(function (result) {
            $scope.disabilitySelections.numeNonCalc = result;
            if ($scope.disabilitySelections.numeNonCalc.DACodes != undefined) {
                $scope.disabilitySelections.numeNonCalcDACodesArray = $scope.disabilitySelections.numeNonCalc.DACodes.split(',');
            }

        }, function (error) {
            $scope.disabilitySelections.numeNonCalc.PRAStudentID = urlsHelper.getLastID($scope.student.Url).toString();
            $scope.disabilitySelections.numeNonCalc.TestID = $scope.tests.numeNonCalc;
            $scope.noNumeracyNonCalc = true;
        });

        function cleanArray(selectedArray, masterDataArray1, masterDataArray2, masterDataArray3, masterDataArray4) {
            var output = [];
            var masterDataArray = masterDataArray1.concat(masterDataArray2);
            masterDataArray = masterDataArray.concat(masterDataArray3);
            masterDataArray = masterDataArray.concat(masterDataArray4);
            angular.forEach(selectedArray, function (item1) {
                angular.forEach(masterDataArray, function (item2) {
                    if (item1 === item2.id) {
                        output.push(item1);                        
                    }
                });
            });
            return output;
            
            
            
        };

        function setArrayItemToTrue(id) {
            var found = false;

            for (var i = 0; i < $scope.student.Disabilities.length; i++) {
                if ($scope.student.Disabilities[i].TestTypeId == id) {
                    found = true;
                    $scope.student.Disabilities[i].HasDisability = true;
                }
            }
            return found;
            
        };
        function setArrayItemToFalse(id) {
            var found = false;

            for (var i = 0; i < $scope.student.Disabilities.length; i++) {
                if ($scope.student.Disabilities[i].TestTypeId == id) {
                    found = true;
                    
                    $scope.student.Disabilities.splice(i, 1);
                }
            }
            return found;

        };
        
        function sendtoWebApi(isEmpty, noEntry, testId, data, from) {
            //$scope.student.Url, $scope.tests.lang, $scope.disabilitySelections.lang
            
            if (isEmpty) {
                if (noEntry) {
                    //do nothing - as it is empty and there was no entry when it was loaded
                } else {
                    // delete this one as it is empty and it has netry when it was loaded
                    studentDisability.deleteStudentDisabilityTypeByURL($scope.student.Url, testId, data);
                }

            } else {
                if (noEntry) {
                    studentDisability.saveStudentDisabilityTypeByURL($scope.student.Url, testId, data);
                } else {
                    studentDisability.updateStudentDisabilityTypeByURL($scope.student.Url, testId, data);
                }
            }
            
        };
        
        

        $scope.cancel = function (disabilitySelections) {
            $modalInstance.dismiss('cancel');
        };
        $scope.submitForm = function (disabilitySelections, $event) {
            $event.preventDefault();
            $event.stopPropagation();
            studentDisability.saveStudentDisabilityCategoryByURL($scope.student.Url, $scope.disabilitySelections.category);
            $scope.disabilitySelections.langDACodesArray = cleanArray($scope.disabilitySelections.langDACodesArray, $scope.DATypesBraille, $scope.DATypesElectronic, $scope.DATypesBlackAndWhite, $scope.DATypes);
            $scope.disabilitySelections.writDACodesArray = cleanArray($scope.disabilitySelections.writDACodesArray, $scope.DATypesBraille, $scope.DATypesElectronic, $scope.DATypesBlackAndWhite, $scope.DATypes);
            $scope.disabilitySelections.readDACodesArray = cleanArray($scope.disabilitySelections.readDACodesArray, $scope.DATypesBraille, $scope.DATypesElectronic, $scope.DATypesBlackAndWhite, $scope.DATypes);
            $scope.disabilitySelections.numeDACodesArray = cleanArray($scope.disabilitySelections.numeDACodesArray, $scope.DATypesBraille, $scope.DATypesElectronic, $scope.DATypesBlackAndWhite, $scope.DATypes);
            $scope.disabilitySelections.numeCalcDACodesArray = cleanArray($scope.disabilitySelections.numeCalcDACodesArray, $scope.DATypesBraille, $scope.DATypesElectronic, $scope.DATypesBlackAndWhite, $scope.DATypes);
            $scope.disabilitySelections.numeNonCalcDACodesArray = cleanArray($scope.disabilitySelections.numeNonCalcDACodesArray, $scope.DATypesBraille, $scope.DATypesElectronic, $scope.DATypesBlackAndWhite, $scope.DATypes);
            $scope.disabilitySelections.lang.DACodes = $scope.disabilitySelections.langDACodesArray.toString();
            $scope.disabilitySelections.writ.DACodes = $scope.disabilitySelections.writDACodesArray.toString();
            $scope.disabilitySelections.read.DACodes = $scope.disabilitySelections.readDACodesArray.toString();
            $scope.disabilitySelections.nume.DACodes = $scope.disabilitySelections.numeDACodesArray.toString();
            $scope.disabilitySelections.numeCalc.DACodes = $scope.disabilitySelections.numeCalcDACodesArray.toString();            
            $scope.disabilitySelections.numeNonCalc.DACodes = $scope.disabilitySelections.numeNonCalcDACodesArray.toString();

            var langEmpty = ($scope.disabilitySelections.lang.DACodes == ""
                &&  $scope.disabilitySelections.lang.AssistiveTechCode == null
                &&  $scope.disabilitySelections.lang.LargePrintCode == null);
            var writEmpty = ($scope.disabilitySelections.writ.DACodes == ""
                &&  $scope.disabilitySelections.writ.AssistiveTechCode == null
                && $scope.disabilitySelections.writ.LargePrintCode == null);
            var readEmpty = ($scope.disabilitySelections.read.DACodes == ""
                &&  $scope.disabilitySelections.read.AssistiveTechCode == null
                && $scope.disabilitySelections.read.LargePrintCode == null);
            var numeEmpty = ($scope.disabilitySelections.nume.DACodes == ""
                &&  $scope.disabilitySelections.nume.AssistiveTechCode == null
                && $scope.disabilitySelections.nume.LargePrintCode == null);
            var numeCalcEmpty = ($scope.disabilitySelections.numeCalc.DACodes == ""
                &&  $scope.disabilitySelections.numeCalc.AssistiveTechCode == null
                && $scope.disabilitySelections.numeCalc.LargePrintCode == null);
            var numeNonCalcEmpty = ($scope.disabilitySelections.numeNonCalc.DACodes == ""
                &&  $scope.disabilitySelections.numeNonCalc.AssistiveTechCode == null
                &&  $scope.disabilitySelections.numeNonCalc.LargePrintCode == null);
                        
            if (langEmpty && writEmpty && readEmpty && numeEmpty && numeCalcEmpty && numeNonCalcEmpty) {
                $scope.student.HasDisability = false;
            } else {
                $scope.student.HasDisability = true;
            }
            
            sendtoWebApi(langEmpty, $scope.noLanguage, $scope.tests.lang, $scope.disabilitySelections.lang, "Language");
            sendtoWebApi(writEmpty, $scope.noWriting, $scope.tests.writ, $scope.disabilitySelections.writ, "Writing");
            sendtoWebApi(readEmpty, $scope.noReading, $scope.tests.read, $scope.disabilitySelections.read, "Reading");
            sendtoWebApi(numeEmpty, $scope.noNumeracy, $scope.tests.nume, $scope.disabilitySelections.nume, "Numeracy");
            sendtoWebApi(numeCalcEmpty, $scope.noNumeracyCalc, $scope.tests.numeCalc, $scope.disabilitySelections.numeCalc, "Numeracy Calc");
            sendtoWebApi(numeNonCalcEmpty, $scope.noNumeracyNonCalc, $scope.tests.numeNonCalc, $scope.disabilitySelections.numeNonCalc, "Numeracy Non Calc");

            
            if (!langEmpty) {
                if (!setArrayItemToTrue($scope.tests.lang)) {
                    $scope.student.Disabilities.push({ TestTypeId: 1, HasDisability: true });
                }
            } else {
                setArrayItemToFalse($scope.tests.lang);
            }
            if (!writEmpty) {                
                if (!setArrayItemToTrue($scope.tests.writ)) {
                    $scope.student.Disabilities.push({ TestTypeId: 2, HasDisability: true });
                }
            } else {
                setArrayItemToFalse($scope.tests.writ);
            }
            if (!readEmpty) {                
                if (!setArrayItemToTrue($scope.tests.read)) {
                    $scope.student.Disabilities.push({ TestTypeId: 3, HasDisability: true });
                }
            } else {
                setArrayItemToFalse($scope.tests.read);
            }
            if ($scope.student.YearLevel == 3 || $scope.student.YearLevel == 5) {
                if (!numeEmpty) {
                    if (!setArrayItemToTrue($scope.tests.nume)) {
                        $scope.student.Disabilities.push({ TestTypeId: 4, HasDisability: true });
                    }
                } else {
                    setArrayItemToFalse($scope.tests.nume);
                }
            }
            if ($scope.student.YearLevel == 7 || $scope.student.YearLevel == 9) { 
                if (!numeCalcEmpty) {                
                    if (!setArrayItemToTrue($scope.tests.numeCalc)) {
                        $scope.student.Disabilities.push({ TestTypeId: 29, HasDisability: true });
                    }
                } else {
                    setArrayItemToFalse($scope.tests.numeCalc);
                }
                if (!numeNonCalcEmpty) {                
                    if (!setArrayItemToTrue($scope.tests.numeNonCalc)) {
                        $scope.student.Disabilities.push({ TestTypeId: 30, HasDisability: true });
                    }
                } else {
                    setArrayItemToFalse($scope.tests.numeNonCalc);
                }
            }
            
            studentData.updateStudent($scope.student.Url, $scope.student);
            _.assign(disabilityStudent, $scope.student);
            $modalInstance.close(disabilitySelections);

            // check to make sure the form is completely valid
            
        };

    }

}
)();
