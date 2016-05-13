import org.apache.axis2.description.AxisOperation

object callSimulCast extends App{
	val ops = new Array[AxisOperation](1)
	val opn: org.apache.axis2.description.OutInAxisOperation = new org.apache.axis2.description.OutInAxisOperation()
	opn.setName(new javax.xml.namespace.QName("http://wmdev02-is8-sandpit.local.fowles.com.au/zzAbhi/Call2", "getSalesForLocation"))
	var str = "Call2\\"
	val svc = new org.apache.axis2.description.AxisService(str)	
	svc.addOperation(opn)
	ops(0)=opn

}