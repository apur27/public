//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, vJAXB 2.1.10 in JDK 6 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2012.12.10 at 10:19:18 AM EST 
//


package au.com.fowles.local.wmdev02_is8_sandpit.zzabhi.call1;

import javax.xml.bind.JAXBElement;
import javax.xml.bind.annotation.XmlElementDecl;
import javax.xml.bind.annotation.XmlRegistry;
import javax.xml.namespace.QName;


/**
 * This object contains factory methods for each 
 * Java content interface and Java element interface 
 * generated in the au.com.fowles.local.wmdev02_is8_sandpit.zzabhi.call1 package. 
 * <p>An ObjectFactory allows you to programatically 
 * construct new instances of the Java representation 
 * for XML content. The Java representation of XML 
 * content can consist of schema derived interfaces 
 * and classes representing the binding of schema 
 * type definitions, element declarations and model 
 * groups.  Factory methods for each of these are 
 * provided in this class.
 * 
 */
@XmlRegistry
public class ObjectFactory {

    private final static QName _GetResponse_QNAME = new QName("http://wmdev02-is8-sandpit.local.fowles.com.au/zzAbhi/Call1", "getResponse");
    private final static QName _Get_QNAME = new QName("http://wmdev02-is8-sandpit.local.fowles.com.au/zzAbhi/Call1", "get");
    private final static QName _GetInputSequence_QNAME = new QName("", "sequence");
    private final static QName _VehicleTypeDoors_QNAME = new QName("", "doors");
    private final static QName _VehicleTypeSalesDesc_QNAME = new QName("", "salesDesc");
    private final static QName _VehicleTypeOdometerUnit_QNAME = new QName("", "odometerUnit");
    private final static QName _VehicleTypeAppletTitle_QNAME = new QName("", "appletTitle");
    private final static QName _VehicleTypeLocation_QNAME = new QName("", "location");
    private final static QName _VehicleTypeComplianceDate_QNAME = new QName("", "complianceDate");
    private final static QName _VehicleTypeRetailEval_QNAME = new QName("", "retailEval");
    private final static QName _VehicleTypeSeats_QNAME = new QName("", "seats");
    private final static QName _VehicleTypeBuildDate_QNAME = new QName("", "buildDate");
    private final static QName _VehicleTypeIntTrim_QNAME = new QName("", "intTrim");
    private final static QName _VehicleTypeVariant_QNAME = new QName("", "variant");

    /**
     * Create a new ObjectFactory that can be used to create new instances of schema derived classes for package: au.com.fowles.local.wmdev02_is8_sandpit.zzabhi.call1
     * 
     */
    public ObjectFactory() {
    }

    /**
     * Create an instance of {@link GetInput }
     * 
     */
    public GetInput createGetInput() {
        return new GetInput();
    }

    /**
     * Create an instance of {@link VehicleType }
     * 
     */
    public VehicleType createVehicleType() {
        return new VehicleType();
    }

    /**
     * Create an instance of {@link VehicleGroup }
     * 
     */
    public VehicleGroup createVehicleGroup() {
        return new VehicleGroup();
    }

    /**
     * Create an instance of {@link VehicleOption }
     * 
     */
    public VehicleOption createVehicleOption() {
        return new VehicleOption();
    }

    /**
     * Create an instance of {@link Get }
     * 
     */
    public Get createGet() {
        return new Get();
    }

    /**
     * Create an instance of {@link GetOutput }
     * 
     */
    public GetOutput createGetOutput() {
        return new GetOutput();
    }

    /**
     * Create an instance of {@link GetResponse }
     * 
     */
    public GetResponse createGetResponse() {
        return new GetResponse();
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link GetResponse }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://wmdev02-is8-sandpit.local.fowles.com.au/zzAbhi/Call1", name = "getResponse")
    public JAXBElement<GetResponse> createGetResponse(GetResponse value) {
        return new JAXBElement<GetResponse>(_GetResponse_QNAME, GetResponse.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link Get }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://wmdev02-is8-sandpit.local.fowles.com.au/zzAbhi/Call1", name = "get")
    public JAXBElement<Get> createGet(Get value) {
        return new JAXBElement<Get>(_Get_QNAME, Get.class, null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "sequence", scope = GetInput.class)
    public JAXBElement<String> createGetInputSequence(String value) {
        return new JAXBElement<String>(_GetInputSequence_QNAME, String.class, GetInput.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "doors", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeDoors(String value) {
        return new JAXBElement<String>(_VehicleTypeDoors_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "salesDesc", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeSalesDesc(String value) {
        return new JAXBElement<String>(_VehicleTypeSalesDesc_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "odometerUnit", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeOdometerUnit(String value) {
        return new JAXBElement<String>(_VehicleTypeOdometerUnit_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "appletTitle", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeAppletTitle(String value) {
        return new JAXBElement<String>(_VehicleTypeAppletTitle_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "location", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeLocation(String value) {
        return new JAXBElement<String>(_VehicleTypeLocation_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "complianceDate", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeComplianceDate(String value) {
        return new JAXBElement<String>(_VehicleTypeComplianceDate_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "retailEval", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeRetailEval(String value) {
        return new JAXBElement<String>(_VehicleTypeRetailEval_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "seats", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeSeats(String value) {
        return new JAXBElement<String>(_VehicleTypeSeats_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "buildDate", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeBuildDate(String value) {
        return new JAXBElement<String>(_VehicleTypeBuildDate_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "intTrim", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeIntTrim(String value) {
        return new JAXBElement<String>(_VehicleTypeIntTrim_QNAME, String.class, VehicleType.class, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link String }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "", name = "variant", scope = VehicleType.class)
    public JAXBElement<String> createVehicleTypeVariant(String value) {
        return new JAXBElement<String>(_VehicleTypeVariant_QNAME, String.class, VehicleType.class, value);
    }

}
