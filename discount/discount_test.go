package main

import (	
	"testing"
	 "math"
)
const float64EqualityThreshold = 1e-9
func AlmostEqual(a, b float64) bool {
    return math.Abs(a - b) <= float64EqualityThreshold
}

func TestDiscountDefault(t *testing.T) {

	expected := 2963.91
	if got := DiscountDefault(3.00, 3.00, 3.00); got != expected {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}
	expected = 767652.69
	if got := DiscountDefault(777.00, 777.00, 777.00); got != expected {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}	
}


func TestDiscountAxilCoffeeRoasters(t *testing.T) {
	expected := 2354.93
	got := DiscountAxilCoffeeRoasters(1.00, 3.00, 3.00)
	if !AlmostEqual(expected, got) {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}
	
}


func TestDiscountMyer(t *testing.T) {
	expected := 982.97
	got := DiscountMyer(1.00, 1.00, 1.00)
	if !AlmostEqual(expected, got) {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}
	
}

func TestGetSecondBiteClassic(t *testing.T) {
	expected := 4
	if got := GetSecondBiteClassic(5); got != expected {
		t.Errorf("expected %d. Got %d instead", expected, got)
	}
	
	expected = 4
	if got := GetSecondBiteClassic(6); got != expected {
		t.Errorf("expected %d. Got %d instead", expected, got)
	}
	
	expected = 5
	if got := GetSecondBiteClassic(7); got != expected {
		t.Errorf("expected %d. Got %d instead", expected, got)
	}
}


func TestGetMyerStandout(t *testing.T) {
	expected := 4
	if got := GetMyerStandout(5); got != expected {
		t.Errorf("expected %d. Got %d instead", expected, got)
	}
	
	expected = 5
	if got := GetMyerStandout(6); got != expected {
		t.Errorf("expected %d. Got %d instead", expected, got)
	}
	
	expected = 6
	if got := GetMyerStandout(7); got != expected {
		t.Errorf("expected %d. Got %d instead", expected, got)
	}
}

func TestApplyPricingRule(t *testing.T) {
	//Default Pricing Rule testing
	expected := 0.00
	got := ApplyPricingRule(0, 0, 0, "default")
	if !AlmostEqual(expected, got) {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}
	expected = 2963.91
	got = ApplyPricingRule(3, 3, 3, "default")
	if !AlmostEqual(expected, got) {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}
	
	expected = 767652.69
	got = ApplyPricingRule(777, 777, 777, "default")
	if !AlmostEqual(expected, got) {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}
	
	//SecondBite Pricing Rule testing
	expected = 0.00
	got = ApplyPricingRule(0, 0, 0, "SecondBite")
	if !AlmostEqual(expected, got) {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}
	
	expected = 2423.93
	got = ApplyPricingRule(1, 3, 3, "SecondBite")
	if !AlmostEqual(expected, got) {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}

	expected = 3233.9
	got = ApplyPricingRule(6, 3, 3, "SecondBite")
	if !AlmostEqual(expected, got) {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}	

	expected = 4043.87
	got = ApplyPricingRule(10, 3, 3, "SecondBite")
	if !AlmostEqual(expected, got) {
		t.Errorf("expected %f. Got %f instead", expected, got)
	}	
	
	//I guess you guess know what i am trying to do here. It is just copy and paste from now on.
	//And there is no case for doing the http Mock testing.
}

















