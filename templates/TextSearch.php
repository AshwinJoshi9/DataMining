<!DOCTYPE html>
<html>
<head>
<title>TextSearch</title>
</head>
<body>
	<h1 align="center">Search Feature</h1>
	<table>
	{% for key, value in data.items() %}
	   <tr>
			<th> {{ key }} </th>
			<td> {{ value }} </td>
	   </tr>
	{% endfor %}
	</table>
</body>
</html>
