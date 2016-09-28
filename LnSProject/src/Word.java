import edu.stanford.nlp.ling.HasWord;


public class Word implements HasWord {

	String word;
	@Override
	public void setWord(String _word) {
		// TODO Auto-generated method stub
		word = _word;
	}

	@Override
	public String word() {
		// TODO Auto-generated method stub
		return word;
	}

}
