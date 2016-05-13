var Component = 
        React.createClass({
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
                return <button onClick={this.update} style={this.buttonStyle}>{this.props.count}</button>
            },
            componentDidMount:function(){
                console.log(this.getDOMNode());
                console.log("Component Did Mount Called!");

                this.increment = setInterval(this.update,1000);
            },
            componentWillUnmount:function(){
                console.log("Component Unmounted Called!");
                clearInterval( this.increment);
            },
    });

 window.renderComponent=function(){
     React.render(<Component count={1}/>,document.getElementById('divContainer'));  
 }

 window.unmountComponent=function(){
     React.unmountComponentAtNode(document.getElementById('divContainer'));  
 }
 
    