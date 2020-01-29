This is a simple go program which runs 1 REST end point whih returns the price of the checkout item.

Once you copy the code in your workspace,all you need to do is run the following command, which will download the dependancy
go get -u "github.com/gorilla/mux"

To run the REST server just fire the command 
go run discount

You should the message on command prompt - 
C:\work\go\ws\src\discount>go run discount
2020/01/29 19:46:06 REST end point Server is running at port 8000

and open a broswer to test all rules with a url like this one - http://localhost:8000/calculate/discount?classic=1&premium=3&standout=10&pricingRules=MYER

To run the unit test just run "go test"


Cheers
