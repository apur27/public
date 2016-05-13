//Example Abhi1
//Creating a class and showing basic HTML on screen with properties.
var Abhi1 = React.createClass({
	render: function() {
		return (
		
			<p className='a1'>
				<br/>Abhisheks React Gallery 1
				
				Disability Categories and Type Selection

					Disability Categories
					
	Select the Categories applicable - {this.props.categories.toString()}
			
			
			
			</p>
		);
	}
	
});
var sdE1=<Abhi1 />;
sdE1.props.categories="Social, Sensory, Cognitive and Physical";
React.render(sdE1, document.getElementById('abhishek1'));

//Example Abhi2
//Initializing a variable 
var Abhi2 = React.createClass({
	render: function() {
		return (
			<p>
			{this.props.copy} Abhisheks React Gallery 2
			
			
			</p>
		);
	}
	
});
var sdE2=<Abhi2 copy={'&copy;'} />
React.render(sdE2, document.getElementById('abhishek2'));

//Example Abhi3
//Initializing a variable with a copyright symbol
var Abhi3 = React.createClass({
	render: function() {
		return (
			<p>
			{this.props.copy} Abhisheks React Gallery 3
			
			
			</p>
		);
	}
	
});
var sdE3=<Abhi3 copy={String.fromCharCode(169)} />
React.render(sdE3, document.getElementById('abhishek3'));


//Example Abhi4
//Initializing a variable in constructor and default props function
//showcasing how the value set in constructor supersedes
//the one set in function getDefaultProps()
//Also there is a concept of state (var1) - which is set using getInitalState()
//and then there is a variable which is either set in getDefaultProps or in constructor
var Abhi4 = React.createClass({
	
	getInitialState: function() {
		return {var1: " I am variable 1"};
	},
	getDefaultProps: function() {
		return {var2: " I am var2 in default props", var3: "I am var3 in default props"};
	},	
	render: function() {
		return (
			<p>Abhisheks React Gallery 4
							<br/>This is var1 ---{this.state.var1} 
							
							<br/>This is var2 ---{this.props.var2} 
							
							<br/>This is var3 ---{this.props.var3} 
			</p>
			
			
		);
	}
	
});
var sdE4=<Abhi4 var2={" I am var2 in constructor"} />
React.render(sdE4, document.getElementById('abhishek4'));

//Example Abhi5
//Basic 2-way data binding from view to a variable (var1)
var Abhi5 = React.createClass({
	
	getInitialState: function() {
		return {var1: " I am variable 1 in initial State"};
	},
	updateVar1:function(element){
		this.setState({var1:element.target.value})
	},	
	render: function() {
		return (
			<p>Abhisheks React Gallery - 5
			Variable Dynmaic binding:<input type="text" placeholder="Place Holder Text" onChange={this.updateVar1} />
				{this.state.var1}
				</p>
		);
	}
	
});
var sdE5=<Abhi5 />
React.render(sdE5, document.getElementById('abhishek5'));

//Example Abhi6
//Updating a variable based on button click and showing it as a label
var Abhi6 = React.createClass({
	update:function(element){
		var clickCount=this.props.count+1;
		this.setProps({count:clickCount})
	},	
	render: function() {
		console.log("Component Render Called - Abhi6");
		return (
			<p>Abhisheks React Gallery 6 -
			<button onClick={this.update}>{this.props.count}</button>
				</p>
		);
	}
	
});
var sdE6=<Abhi6 count={0}/>
React.render(sdE6, document.getElementById('abhishek6'));


//Example Abhi7
//Demonstration of componentWillMount, componentDidMount and render function
//and the order in which they are fired.
var Abhi7 = React.createClass({
	update:function(element){
		var clickCount=this.props.count+1;
		this.setProps({count:clickCount})
	},		
	componentWillMount:function() {
		console.log("Component Will now Mount called! - Abhi7");
	},
	componentDidMount:function() {
		console.log("Component did Mount called! - Abhi7");
	},

	render: function() {
		console.log("Component Render Called - Abhi7");
		return (
			<p>Abhisheks React Gallery 7		
			<button onClick={this.update}>{this.props.count}</button>
				</p>
		);
	}
	
});
var sdE7=<Abhi7 count={0}/>
React.render(sdE7, document.getElementById('abhishek7'));


//Example Abhi8
//Demonstration of unmountComponentAtNode and render function
//uisng HTML functions calls from button.
var Abhi8 = React.createClass({
	update:function(element){
		var clickCount=this.props.count+1;
		this.setProps({count:clickCount})
	},		
	componentWillMount:function() {
		console.log("Component Will now Mount called! - Abhi8");
	},
	componentDidMount:function() {
		console.log("Component did Mount called! - Abhi8");
	},
	componentWillUnmount:function() {
		console.log("Component Unmounted called! - Abhi8");
	},
	render: function() {
		console.log("Component Render Called - Abhi8");
		return (
			<p>Abhisheks React Gallery 8		
			<button onClick={this.update}>{this.props.count}</button>
				</p>
		);
	}
	
});
window.renderComponent8=function(){
		var sdE8=<Abhi8 count={0}/>
		React.render(sdE8, document.getElementById('abhishek8'));
}

window.unmountComponent8=function(){		
		React.unmountComponentAtNode(document.getElementById('abhishek8'));
}




//Example Abhi9
//Demonstration of unmountComponentAtNode and render function,
//along with rest of class variable.
//uisng HTML functions calls from button.
var Abhi9 = React.createClass({
	update:function(element){
		var clickCount=this.props.count+1;
		this.setProps({count:clickCount})
	},		
	componentWillMount:function() {
		console.log("Component Will now Mount called! - Abhi9");
	},
	componentDidMount:function() {
		console.log(this.getDOMNode());
		console.log("Component did Mount called! - Abhi9");
		this.increment=setInterval(this.update,1000);
	},
	componentWillUnmount:function() {
		console.log("Component Unmounted called! - Abhi9");
		clearInterval(this.increment);
	},
	render: function() {
		console.log("Component Render Called - Abhi9");
		return (
			<p>Abhisheks React Gallery 9		
			<button onClick={this.update}>{this.props.count}</button>
				</p>
		);
	}
	
});
window.renderComponent9=function(){
		var sdE9=<Abhi9 count={0}/>
		React.render(sdE9, document.getElementById('abhishek9'));
}

window.unmountComponent9=function(){		
		React.unmountComponentAtNode(document.getElementById('abhishek9'));
}



//Example Abhi10
//Demonstration of componentWillMount and render function,

var Abhi10 = React.createClass({
	update:function(element){
		var clickCount=this.props.count+1;
		this.setProps({count:clickCount})
	},		
	componentWillMount:function() {
		console.log("Component Will now Mount called! - Abhi10");
		if(this.props.count===1) {
			this.buttonStyle={'color':'green'}
		}
	},
	componentDidMount:function() {
		console.log("Component did Mount called! - Abhi10");
	},
	componentWillUnmount:function() {
		console.log("Component Unmounted called! - Abhi10");
	},
	render: function() {
		console.log("Component Render Called - Abhi10");
		return (
			<p>Abhisheks React Gallery 10		
			<button onClick={this.update} style={this.buttonStyle}>{this.props.count}</button>
				</p>
		);
	}
	
});
var sdE10=<Abhi10 count={1}/>
React.render(sdE10, document.getElementById('abhishek10'));




//Example Abhi11
//Demonstration of componentWillMount and render function,
//along with componentWillReceiveProps

var Abhi11 = React.createClass({
	update:function(element){
		var clickCount=this.props.count+1;
		this.setProps({count:clickCount})
	},		
	componentWillMount:function() {
		console.log("Component Will now Mount called! - Abhi11");
		if(this.props.count===1) {
			this.buttonStyle={'color':'green'}
		}
	},
	componentDidMount:function() {
		console.log("Component did Mount called! - Abhi11");
	},
	componentWillUnmount:function() {
		console.log("Component Unmounted called! - Abhi11");
	},
	componentWillReceiveProps:function(nextProp) {
		console.log(nextProp);
		console.log("Component Will Receive Props called! - Abhi11");
	},	
	render: function() {
		console.log("Component Render Called - Abhi11");
		return (
			<p>Abhisheks React Gallery 11		
			<button onClick={this.update} style={this.buttonStyle}>{this.props.count}</button>
				</p>
		);
	}
	
});
var sdE11=<Abhi11 count={1}/>
React.render(sdE11, document.getElementById('abhishek11'));

//Example Abhi12
//Demonstration of componentWillMount and render function,
//along with componentWillReceiveProps and shouldComponentUpdate
//and their order
//shouldComponentUpdate decide whether render will be called or not.
var Abhi12 = React.createClass({
	getInitialState: function() {
		return {updation:false};
	},
	update:function(element){
		var clickCount=this.props.count+1;
		this.setProps({count:clickCount})
	},		
	componentWillMount:function() {
		console.log("Component Will now Mount called! - Abhi12");
		if(this.props.count===1) {
			this.buttonStyle={'color':'green'}
		}
	},
	componentDidMount:function() {
		console.log("Component did Mount called! - Abhi12");
	},
	componentWillUnmount:function() {
		console.log("Component Unmounted called! - Abhi12");
	},
	componentWillReceiveProps:function(nextProps) {
		//console.log(nextProp);
		console.log("Component Will Receive Props called! - Abhi12");
		this.setState ({updation:nextProps.count>this.props.count});
	},	

	render: function() {
		console.log("Component Render Called - Abhi12");
		console.log(this.state.updation);
		return (
			<p>Abhisheks React Gallery 12		
			<button onClick={this.update} style={this.buttonStyle}>{this.props.count}</button>
				</p>
		);
	},
	shouldComponentUpdate : function(nextProps, nextState) {
		console.log("souldComponentUpdate is called! - Abhi12");
		console.log(nextProps);
		console.log(nextState);
		return nextProps.count % 2 === 0;
		//return false;
	},		
	componentWillUpdate : function(nextProps, nextState) {
		console.log("componentWillUpdate is called! - Abhi12");
		console.log(nextProps);
		console.log(nextState);		
	},			
	componentDidUpdate : function(nextProps, nextState) {
		console.log("componentDidUpdate is called! - Abhi12");
		console.log(nextProps);
		console.log(nextState);		
		console.log(this.getDOMNode());		
	},				
});

window.renderComponent12=function(){
		var sdE12=<Abhi12 count={1}/>
		React.render(sdE12, document.getElementById('abhishek12'));
}

window.unmountComponent12=function(){		
		React.unmountComponentAtNode(document.getElementById('abhishek12'));
}

//Example Abhi13
//Demonstration of multiple classes
//
var Movie = React.createClass({
	render: function() {
		return (
			<div>	
				{this.props.movieName}					
			</div>
		);
	}
	
});
var AbhiMovies13 = React.createClass({
	render: function() {
		return (
		<p>Abhisheks React Gallery 13 
		
				<br/>I have watched the following movies
					
					<br/><Movie movieName="The Shawshank Redemption" />
					<br/><Movie movieName="Last of the Mohicons" />
					<br/><Movie movieName="The usual suspects" />
					<br/><Movie movieName="Bad Boys 2" />
					<br/><Movie movieName="The skeleton key" />					
		</p>			
			
			
		);
	}
	
});

var sdE13=<AbhiMovies13 />;
React.render(sdE13, document.getElementById('abhishek13'));

//Example Abhi14
//Demonstration of multiple classes
//and use of external variable 

var movieList = [
					{movieName:	"The Shawshank Redemption"},
					{movieName:	"Last of the Mohicons"},
					{movieName:	"The usual suspects"},
					{movieName:	"Bad Boys 2"},
					{movieName:	"The usual suspects"},
					{movieName:	"The skeleton key"}

];
var Movie = React.createClass({
	render: function() {
		return (
			<div>	
				{this.props.movieName}					
			</div>
		);
	}
	
});
var AbhiMovies14 = React.createClass({
	render: function() {
		
		var movie=this.props.list.map(function(movie) {
			return <Movie movieName={movie.movieName} />
		});
		
		return (
		<p>Abhisheks React Gallery 14 
		
				<br/>I have watched the following movies					
					<br>{movie}</br>
		</p>			
			
			
		);
	}
	
});

var sdE14=<AbhiMovies14 list={movieList}/>;
React.render(sdE14, document.getElementById('abhishek14'));


//Example Abhi15
//Demonstration of multiple classes
//and use of external variable
//new variation

var movieList = [
					{movieName:	"The Shawshank Redemption"},
					{movieName:	"Last of the Mohicons"},
					{movieName:	"The usual suspects"},
					{movieName:	"Bad Boys 2"},
					{movieName:	"The usual suspects"},
					{movieName:	"The skeleton key"}

];
var Movie = React.createClass({
	render: function() {
		return (
			<div>	
				{this.props.movieName}					
			</div>
		);
	}
	
});
var AbhiMovies15 = React.createClass({
	render: function() {
		
		var movie=this.props.list.map(function(movie) {
			return <Movie movieName={movie.movieName} />
		});
		
		return (<div>
		Abhisheks React Gallery 15 
		
				<br/>I have watched the following movies					
					<br><MovieCollection list={this.props.list} /></br>
		
			</div>
			
		);
	}
	
});

var MovieCollection = React.createClass({
	render: function() {
		
		var movie=this.props.list.map(function(movie) {
			return <Movie movieName={movie.movieName} />
		});
		
		return (<div> {movie} </div>);
	}
	
});

var sdE15=<AbhiMovies15 list={movieList}/>;
React.render(sdE15, document.getElementById('abhishek15'));


//Example Abhi16
//Re-usable components
//
//

var movieList1 = [
					{movieName:	"The Shawshank Redemption"},
					{movieName:	"Last of the Mohicons"},
					{movieName:	"The usual suspects"},
					{movieName:	"Bad Boys 2"},
					{movieName:	"The usual suspects"},
					{movieName:	"The skeleton key"}

];

var ReviewControl16 = React.createClass({
	getInitialState: function() {
		return {name:'', feedback:'', course:'', reviews: []};
	},
	onChangeName:function(e1){
		this.setState({name:e1.target.value})
	},
	onChangeFeedback:function(e1){
		this.setState({feedback:e1.target.value})
	},		
	onChangeCourse:function(e1){
		this.setState({course:e1.target.value})
	},				
	submitReview:function(e1){
		e1.preventDefault();
		this.state.reviews.push({name:this.state.name, feedback:this.state.feedback, course:this.state.course});
		this.setState({name:'', feedback:''});
	},						

	render: function() {

		var movies=this.props.list.map(function(movie) {
			return <option key={movie.movieName} value={movie.movieName}>{movie.movieName}</option>
		});
		
		return (<div>
		Abhisheks React Gallery 16 
					<form onSubmit={this.submitReview}>
						<label> Name </label>
						<input type="text" placeholder="Enter Your Name" value={this.state.name} onChange={this.onChangeName} />
						<br/><br/>
						<label> Feedback </label>						
						<input type="text" placeholder="Enter Your Feedback" value={this.state.feedback} onChange={this.onChangeFeedback} />
						<br/><br/>
						<select onChange={this.onChangeCourse}>
							{movies}
						</select>
						<br/><br/>
						<input type="Submit" value="Submit" />
					</form>
					<ReviewCollection reviews={this.state.reviews} />
			</div>
			
		);
	}
	
});

var ReviewCollection = React.createClass({
	render: function() {
		
		var reviews=this.props.reviews.map(function(review) {
			return <Review course={review.course} name={review.name} feedback={review.feedback} />
		});
		
		return (<div> {reviews} </div>);
	}
	
});


var Review = React.createClass({
	render: function() {		
		
		return (
		<div> 
			<span>Name</span> {this.props.name}
			<br/>
			<span>Movie</span> {this.props.course}
			<br/>
			<span>Feedback</span> {this.props.feedback}
		</div>);
	}
	
});

var sdE16=<ReviewControl16 list={movieList1}/>;
React.render(sdE16, document.getElementById('abhishek16'));












