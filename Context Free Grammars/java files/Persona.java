import java.util.List;
public class Persona {
	//Attributes
	public String nombre;
	public int edad;
	public float altura;

	//Methods
	public String hablar() { return null; }
	public void caminar() {  }
	public void conducir(String carro, int velocidad) {  }

	//Relations
	private List<Carro> carro;
	private Licencia licencia;
	private Conductor conductor;
	private List<Empresa> empresa;
}
