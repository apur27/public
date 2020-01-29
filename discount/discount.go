package main

import (	
		"fmt"
		"log"
		"net/http"					
		"github.com/gorilla/mux"
        "strconv"
)
const DEFAULT = "default"
const SECOND_BITE = "SecondBite"
const AXIL_COFFEE_ROASTERS = "Axil Coffee Roasters"
const MYER = "MYER"

const DEFAULT_CLASSIC_PRICE float64  = 269.99
const DEFAULT_STANDOUT_PRICE float64  = 322.99
const DEFAULT_PREMIUM_PRICE float64  = 394.99

const AXIL_COFFEE_ROASTERS_STANDOUT_PRICE float64  = 299.99

const MYER_PREMIUM_PRICE float64  = 389.99


type checkout struct {
        Classic 		int
		Premium 		int
		Standout		int
		PricingRules	string
}

func ParseInt(s string, dest interface{}) error {
        d, ok := dest.(*int)
        if !ok {
			return fmt.Errorf("wrong type for ParseInt: %T", dest)
        }
        n, err := strconv.Atoi(s)
        if err != nil {
			return err
        }		
        *d = n
        return nil
}

func DiscountDefault(classic, premium, standout float64) float64  {
		return DEFAULT_CLASSIC_PRICE*classic+DEFAULT_STANDOUT_PRICE*standout+DEFAULT_PREMIUM_PRICE*premium
}
func DiscountAxilCoffeeRoasters(classic, premium, standout float64) float64  {
		return DEFAULT_CLASSIC_PRICE*classic+AXIL_COFFEE_ROASTERS_STANDOUT_PRICE*standout+DEFAULT_PREMIUM_PRICE*premium
}
func DiscountMyer(classic, premium, standout float64) float64  {
		return DEFAULT_CLASSIC_PRICE*classic+DEFAULT_STANDOUT_PRICE*standout+MYER_PREMIUM_PRICE*premium
}
func GetSecondBiteClassic(classic int) int  {
		return (classic/3)*2+(classic%3)
}
func GetMyerStandout(standout int) int  {
		return (standout/5)*4+(standout%5)
}
func ApplyPricingRule(classic, premium, standout int, pricingRules string) float64  {
        switch pricingRules {
			case DEFAULT: return DiscountDefault(float64(classic), float64(premium), float64(standout))
			case SECOND_BITE: return DiscountDefault(float64(GetSecondBiteClassic(classic)), float64(premium), float64(standout))
			case AXIL_COFFEE_ROASTERS: return DiscountAxilCoffeeRoasters(float64(classic), float64(premium), float64(standout))
			case MYER: return DiscountMyer(float64(classic), float64(premium), float64(GetMyerStandout(standout)))
			default: panic("illegal Pricing Rules")
		}
        return 0
}
func main() {
	router := mux.NewRouter()
	router.HandleFunc("/calculate/discount", GetPrice).Methods("GET")
	log.Println("REST end point Server is running at port 8000")
	log.Fatal(http.ListenAndServe(":8000", router))
}

func GetPrice(writer http.ResponseWriter, request *http.Request) {
	log.Println("Calculate discount")
	if err := request.ParseForm(); err != nil {
			log.Printf("Error parsing form: %s", err)
			return
	}
	chk := &checkout{}
	if err := ParseInt(request.Form.Get("classic"), &chk.Classic); err != nil {
			log.Printf("Error parsing Classic: %s", err)
			return
	}
	if err := ParseInt(request.Form.Get("premium"), &chk.Premium); err != nil {
			log.Printf("Error parsing Premium: %s", err)
			return
	}
	if err := ParseInt(request.Form.Get("standout"), &chk.Standout); err != nil {
			log.Printf("Error parsing Standout: %s", err)
			return
	}	
	chk.PricingRules = request.Form.Get("pricingRules")
	fmt.Fprintf(writer, "Checkout Data Classic: %d, Premium: %d, Standout: %d, PricingRules: %s\n", chk.Classic, chk.Premium, chk.Standout, chk.PricingRules)
	fmt.Fprintf(writer, "Final Price: %.2f\n", ApplyPricingRule(chk.Classic, chk.Premium, chk.Standout, chk.PricingRules))
	 
	
}