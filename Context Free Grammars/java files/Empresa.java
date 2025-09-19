import java.util.List;
public class Empresa {
	//Attributes
	public String nombre;
	public long nit;

	//Methods
	public void contratar(String persona) {  }
	public void asignarVehiculo(String carro) {  }

	//Relations
	private List<Persona> persona;
	private List<Carro> carro;
}
