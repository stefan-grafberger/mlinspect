setTimeout(function(){
   document.querySelectorAll('#pipeline-textarea').forEach(block => {
        // then highlight each
        var editor = CodeMirror.fromTextArea(block, {
            lineNumbers: true,
            mode: "text/x-python"
        });
    });

}, 3000);