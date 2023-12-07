use native_dialog::{FileDialog, MessageDialog, MessageType};

fn main() {
    let path = FileDialog::new()
        .set_location("~")
        .set_title("asd test title")
        //.add_filter("PNG Image", &["png"])
        //.add_filter("JPEG Image", &["jpg", "jpeg"])
        //.show_open_single_file()
        .show_open_single_dir()
        .unwrap();

    let path = match path {
        Some(path) => {
            println!("{}", path.display());
            path
        }
        None => return,
    };

    // TODO: confirmation dialog is not shown
    let yes = MessageDialog::new()
        .set_type(MessageType::Info)
        .set_title("Do you want to open the file?")
        .set_text(&format!("{:#?}", path))
        .show_confirm()
        .unwrap();

    if yes {
        println!("yes")
    }
}
