setTimeout(function(){
   document.querySelectorAll('#pipeline-textarea').forEach(block => {

        // then highlight each
        var editor = CodeMirror.fromTextArea(block, {
            lineNumbers: true,
            matchBrackets: true,
            mode: "text/x-python",
            theme : "default"
            // can use other themes, only need to include css file from https://cdnjs.com/libraries/codemirror
            // see https://codemirror.net/demo/theme.html#default
        });

        document.addEventListener("keydown", keyDownTextField, false);
        function keyDownTextField(e) {
        var keyCode = e.keyCode;
          if(keyCode==13) { // Enter key
            editor.markText({line:1,ch:0},{line:10,ch:0},{className:  "code-dag-highlight"});
          }
        }
    });

}, 2000);