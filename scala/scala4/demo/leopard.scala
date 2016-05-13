object leopard extends App{  
    try {
      var number1 = (1 to 45).toList            
      val f = new java.io.File("c:\\scala3\\leopard\\out.txt")
      val out = new java.io.PrintWriter( f )
      number1 = ghumao(util.Random.nextInt(99), number1)
      for (jk <- 0 to 2) {
	      number1 = ghumao(util.Random.nextInt(35), number1)
	      number1 = ghumao(util.Random.nextInt(35), number1)	      
	      var (number11, number2) = number1.splitAt(15);
	      var (number12, number13) = number2.splitAt(15);
	      lagao (number11, out)
	      lagao (number12, out)
	      lagao (number13, out)
	      
      }
       out.close
    } catch {
      case e: Exception => 
        println("Usage: yabba dabba do ... "+ e)
    }
    def ghumao(n: Int, number1: List[Int]):  List[Int]= {    
    	var number2 = number1
    	val tp = List(102,122,143,122,2,3,7,6,9,10,19,21,36,34,45,16,133,309,199,187,139,3769,99,149)    	
    	var main = util.Random.shuffle(tp)
    	def process(number1: List[Int]): List[Int]  = {
    			main = util.Random.shuffle(main) 
    			val ak = util.Random.nextInt(main(0))
    			println(ak)
    			val bk = util.Random.nextInt(main(1))
    			println(bk)
    			val rk = util.Random.nextInt(main(2))
    			println(rk)
    			
    			
	    		for( a <- 1 to bk) {for( b <- 1 to rk) {for( c <- 1 to ak)  {
	    						number2 = util.Random.shuffle(number2)}}    			
	    						main = util.Random.shuffle(main)
	    					}
    			number2
    	}
    	if (n == 0) process(number1)
    		else ghumao(n-1, process(number1))
    		
    }
    
    def lagao (xman: List[Int], out: java.io.PrintWriter) = {
    	var xmanT = xman
	      for (a <- 0 to 2) {
		   xmanT = ghumao(util.Random.nextInt(21), xmanT) 
		   xmanT = ghumao(util.Random.nextInt(21), xmanT)       
		   xmanT = ghumao(util.Random.nextInt(21), xmanT)       

			for (b <- 0 to 5)       	{
				out.print((xmanT(b)).toString())
				out.print("    ")      			
			}    		
		    out.println("    ")
	      }

    	
    	
    }
  
}

