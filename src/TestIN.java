import net.sf.jsqlparser.expression.operators.relational.InExpression;
import java.lang.reflect.Method;

public class TestIN {
    public static void main(String[] args) {
        Method[] methods = InExpression.class.getMethods();
        for (Method m : methods) {
            System.out.println(m.getName() + " -> " + m.getReturnType().getName());
        }
    }
}
