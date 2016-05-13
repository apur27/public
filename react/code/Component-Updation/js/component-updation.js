var Component = 
        React.createClass({
            getInitialState:function(){
                return {updation:false}
            },
            update:function(){
                var clickCount = this.props.count+1;
                this.setProps({count:clickCount});
            },
            componentWillMount:function(){
                console.log("Component Will Mount Called!");
                if(this.props.count===1){
                    this.buttonStyle={'color':'green'}
                }
              
            },
            render:function(){
                console.log("Component Render Called!");
                console.log(this.state.updation);
                return <button onClick={this.update} style={this.buttonStyle}>{this.props.count}</button>
            },
            componentDidMount:function(){
                console.log(this.getDOMNode());
                console.log("Component Did Mount Called!");

                //this.increment = setInterval(this.update,1000);
            },
            componentWillUnmount:function(){
                console.log("Component Unmounted Called!");
                //clearInterval( this.increment);
            },
            componentWillReceiveProps:function(nextProps) {
                //console.log(nextProps);
                console.log("componentWillReceiveProps is called");
                this.setState({updation:nextProps.count>this.props.count})
            },
            shouldComponentUpdate : function(nextProps,nextState){
                console.log("shouldComponentUpdate is called");
                console.log(nextProps);
                console.log(nextState);
                return nextProps.count % 2 === 0;
                //return false;
            },
            componentWillUpdate: function(nextProps, nextState) {
              console.log("componentWillUpdate is called");
              console.log(nextProps);
              console.log(nextState);
            },
            componentDidUpdate: function(prevProps, prevState) {
               console.log("componentDidUpdate is called");
               console.log(prevProps);
               console.log(prevState);
               console.log(this.getDOMNode());
            }


    });

 window.renderComponent=function(){
     React.render(<Component count={1}/>,document.getElementById('divContainer'));  
 }

 window.unmountComponent=function(){
     React.unmountComponentAtNode(document.getElementById('divContainer'));  
 }
 